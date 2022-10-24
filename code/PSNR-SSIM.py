import os
import cv2
from skimage import metrics
import time


def calc_measures(hr_path, fake_path, name, file_name, calc_psnr=True, calc_ssim=True):
    HR_files = os.listdir(hr_path)
    mean_psnr = 0
    mean_ssim = 0
    num = 0

    for file in HR_files:
        path_hr = os.path.join(hr_path, file)
        hr_img = cv2.imread(path_hr, 1)
        path = os.path.join(fake_path, file)
        # print(path)
        if not os.path.isfile(path):
            raise FileNotFoundError('')
        # print(path)
        inf_img = cv2.imread(path, 1)
        # print(inf_img)
        num += 1
        if num % 100 == 0:
            print(num)

        if calc_psnr:
            psnr = metrics.peak_signal_noise_ratio(hr_img, inf_img)
            mean_psnr += psnr
        if calc_ssim:
            ssim = metrics.structural_similarity(hr_img, inf_img, multichannel=True)
            mean_ssim += ssim

    print('-' * 10)
    if calc_psnr:
        M_psnr = mean_psnr / len(HR_files)
        print('mean-PSNR %f dB' % M_psnr)
    if calc_ssim:
        M_ssim = mean_ssim / len(HR_files)
        print('mean-SSIM %f' % M_ssim)

    txt_file = open(file_name, 'a+')
    txt_file.write(name)
    txt_file.write('\n')
    txt_file.write(str(time.asctime(time.localtime(time.time()))))
    txt_file.write('\n')
    txt_file.write('mean-PSNR: %f , mean-SSIM: %f' % (M_psnr, M_ssim))
    txt_file.write('\n' * 2)


if __name__ == '__main__':
    for i in range(9, 27):  # birds flowers celeba15000
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(str(i*10000))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        calc_measures('../real/real_flower',
                      '../images/flower/only' + str(i*10000),
                      str(i*10000), '../result/flower/psnr_ssim.txt',
                      calc_psnr=True, calc_ssim=True)
