from PIL import Image
import pytesseract
import argparse
import cv2
import os
from config import config, log_config
from difflib import SequenceMatcher

def getText(image):
    gray = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imwrite("bin.jpg",gray)
    gray = Image.fromarray(gray.astype('uint8'))
    text = pytesseract.image_to_string(gray)
    return text

def compare(str1, str2):
    ratio = SequenceMatcher(None, str1, str2).ratio()
    return ratio
    
def getAccuracy(res, hr, lr, bic, i):
    ocrres = open("ocrresults.txt", "a")
    res = res[:,:,0]
    hr = hr[:,:,0]
    lr = lr[:,:,0]
    bic = bic[:,:,0]
    if i==0:
        ocrres.write('='*50+'\n')
    ann_file_path = config.VALID.annot_path+"test-annot-"+format(i+1, '04d')+".txt"
    ann_file = open(ann_file_path, 'r')
    ann_text = ann_file.read().replace('_', ' ')
    ann_file.close()
    
    res_text = getText(res)
    hr_text = getText(hr)
    lr_text = getText(lr)
    bic_text = getText(bic)
    #using HR as baseline for accuracy
    hr_acc = compare(ann_text, hr_text)
    res_acc = compare(ann_text, res_text)
    lr_acc = compare(ann_text, lr_text)
    bic_acc = compare(ann_text, bic_text)
    ocrres.write('LRI: '+str(lr_acc)[:8]+'\t'+lr_text+'\n'
                 +'BIC: '+str(bic_acc)[:8]+'\t'+bic_text+'\n'
                 +'GEN: '+str(res_acc)[:8]+'\t'+res_text+'\n'
                 +'HRI: '+str(hr_acc)[:8]+'\t'+hr_text+'\n\n')
    ocrres.close()
    return res_acc, hr_acc, lr_acc, bic_acc
    
    #return 0
    #print(text)
    
#getAccuracy(0,0,54)
