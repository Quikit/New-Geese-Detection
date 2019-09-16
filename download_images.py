import urllib.request
import cv2 as cv
import numpy as np
import os

def store_raw_imgs(link, start=1):
    goose_img = cv.imread('goose.jpg')
    n_goose_img = cv.resize(goose_img, (50, 50))
    cv.imwrite('resized_goose.jpg', n_goose_img)
    neg_images_link = link   
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = start
    
    if not os.path.exists('neg'):
        os.makedirs('neg')
    
    print('Creating Negatives from ImageNet...')   
    for i in neg_image_urls.split('\n'):
        
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            img = cv.imread("neg/"+str(pic_num)+".jpg",cv.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv.resize(img, (100, 100))
            cv.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))
            
    return pic_num
            

def find_uglies():
    match = False
    
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv.imread('uglies/'+str(ugly))
                    question = cv.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print(f'Deleting... {current_image_path}')
                        os.remove(current_image_path)
                        
                except Exception as e:
                    print(str(e))
 
 
def create_pos_n_neg():
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            '''
            if file_type == 'images/pos':
                line = file_type+'/'+img+' 1 0 0 100 100\n'
                with open('info.dat','a') as f:
                    f.write(line)
            '''
                    
            if file_type == 'neg':
                line = file_type + '/' + img + '\n'
                
                with open('bg.txt','a') as f:
                    f.write(line)
                    
link1 = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04409515'
link2 = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02802426'
link3 = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04081281'
link4 = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00017222'
              
# stopped_pic_num = store_raw_imgs(link3, 856)
# store_raw_imgs(link4, stopped_pic_num)
find_uglies()
create_pos_n_neg()
