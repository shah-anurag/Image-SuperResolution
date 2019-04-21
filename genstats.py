def printArray(title, array):
    print(title+" values")
    for a in array:
        print(a)
    print('\n')

interlog = open('inter_eval.txt', 'r')
lines = []
for line in interlog:
    l = line.strip()
    if(len(l>1)):
        lines.append(line)
lines.close()
psnr = []
mse = []
ssim = []
imp = []
for line in lines:
    values = line.split(':')
    psnr.append(values[1][1:9])
    mse.append(values[2][1:9])
    ssim.append(values[3][1:9])
    imp.append(values[4][1:])
printArray("PSNR", psnr)
printArray("MSE", mse)
printArray("SSIM", ssim)
printArray("Improvement", imp)
