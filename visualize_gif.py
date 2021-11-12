from PIL import Image
import glob
from visualize import plot
from genotypes import Genotype


def resize_img(img, width, height):
    # 作りたいサイズの白紙画像を生成
    new_img = Image.new(img.mode, (width, height), (255, 255, 255))

    # 白紙画像に元画像をペースト
    left_loc = (width / 2) - (img.size[0] / 2)
    top_loc = (height / 2) - (img.size[1] / 2)
    new_img.paste(img, (int(left_loc), int(top_loc)))

    return new_img


# logファイルから各エポックのGenotypeを取得
def get_genotypes(log_path):
    with open(log_path + '/log.txt') as f:
        raw_lines = [line.strip() for line in f.readlines()]

        genotypes = []
        for i in range(7):
            genotypes.append(eval(raw_lines[4 + i * 12][35:]))

    return genotypes


def create_gif(img_path, output_name):
    files = sorted(glob.glob(img_path + '/*.png'))
    images = list(map(lambda file: Image.open(file), files))
    resized_images = [resize_img(image, 1500, 600) for image in images]
    resized_images[0].save(output_name, save_all=True, append_images=resized_images[1:], duration=300, loop=0)


if __name__ == '__main__':
    genos = get_genotypes('./logs/search-EXP-20211112-113450')

    # Genotypeからpngを生成（各エポックに対して）
    for i, geno in enumerate(genos):
        plot(geno.normal, f'gif/normal/epoch{i + 1}', False)
        plot(geno.reduce, f'gif/reduction/epoch{i + 1}', False)

    # gifの生成
    create_gif('gif/normal', 'gif/normal.gif')
    create_gif('gif/reduction', 'gif/reduction.gif')
