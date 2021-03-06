import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        # for deeperDARTS
        self.sn_width = args.arch_sn_width
        self.r_rate = 1
        self.reg = 0

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        """
            引数： {
                input_train: trainデータ(input),
                target_train: trainデータ(target),
                input_valid: validデータ(input),
                target_valid: validデータ(target),
                eta: 学習率（lr）,
                network_optimizer: 重み(ω)のoptimizer,
                unrolled: second_orderにするか否か,
            }
        """

        # アーキテクチャパラメータ(α)のoptimizerをzero_grad
        self.optimizer.zero_grad()

        # second order
        if unrolled:
            # ∂Lval(ω - lr * [∂Ltrain(ω,α) / ∂ω],α) / ∂α
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        # first order
        else:
            # ∂Lval(ω,α) / ∂α
            self._backward_step(input_valid, target_valid)

        # アーキテクチャパラメータ(α)のoptimizerをstep
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        real_loss = self.model._loss(input_valid, target_valid)
        self.reg = self._compute_reg(self.model)

        loss = real_loss + self.reg * self.r_rate
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        # ω’ = ω - lr * [∂Ltrain(ω,α) / ∂ω]　に更新したmodelの作成
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)

        # loss: Lval(ω - lr * [∂Ltrain(ω,α) / ∂ω],α) と　正則化項 の計算
        real_loss = unrolled_model._loss(input_valid, target_valid)
        self.reg = self._compute_reg(unrolled_model)

        # loss と 正則化項　の和
        unrolled_loss = real_loss + self.reg * self.r_rate

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig.data, alpha=eta)

        # unrolled_modelのαから，元のαへコピー
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        # trainデータでのLoss計算
        loss = self.model._loss(input, target)

        # 現在の重み(ω)をconcat
        theta = _concat(self.model.parameters()).data

        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)

        # (dL/dw + weight_decay * w) の計算 (weight decayの適用）
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta

        # パラメータを，w - eta * (moment + dtheta) に設定したmodelの作成
        unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))

        return unrolled_model

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        # modelのパラメータ(theta)を，データ整形しつつ，paramsに代入していく
        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())  # np.prob: すべての要素の積
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()  # r / L2ノルム
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=(2 * R))
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    # 正則化項の計算
    def _compute_reg(self, model):
        # αの各行がどのnodeからのedgeに対応しているかのlist（例: edgeが[prev_prev -> node1] => start_node=1）
        # list_start_node = [1, 1, 1, 1, 1.3, 1, 1, 1.3, 1.6, 1, 1, 1.3, 1.6, 1.9]  # 幅0.3
        n_in_node = 2
        list_start_node = []
        for i in range(self.model.steps):
            for j in range(i + n_in_node):
                if j >= 2:
                    list_start_node.append(1 + self.sn_width * (j - 1))
                else:
                    list_start_node.append(1)

        # a^2のとき
        # alphas_normal_squared = torch.pow(F.softmax(model.alphas_normal, dim=-1), 2)

        reg = 0
        for start_node, params_row in zip(list_start_node, torch.sigmoid(model.alphas_normal)):
            reg += torch.sum(params_row / start_node)
        return reg
