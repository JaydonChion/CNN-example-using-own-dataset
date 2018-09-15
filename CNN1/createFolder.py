import os
import random as random
import shutil

folder_path = "/Users/jetherngchion/Downloads/all-2"  #path of the main folder
current_path = folder_path + "/train"               #path of your dataset
def create_dataset(name, ratio, shuffle = True):        #function to split into train / test /validation
    
    data = os.listdir(current_path)
    if(shuffle):
        random.shuffle(data)
    new_path = folder_path + "/"+name
    os.mkdir(new_path)
    file_to_be_moved = data[:int(len(data)*ratio)]
    new_path = folder_path + "/"+name
    for i in range(len(file_to_be_moved)):
        file = random.choice(os.listdir(current_path))
        file_path = current_path + "/"+file
        shutil.move(file_path,new_path)


def sort_category(name):                    #sort the images into different categories
    current_path = folder_path +'/'+name
    os.chdir(current_path)
    os.mkdir(current_path +'/'+'dog')
    os.mkdir(current_path +'/'+'cat')

    for i in os.listdir(current_path):
       
        if(i[-3:] == "jpg"):
            if(i[:3] == "dog"):
                shutil.move(current_path +'/' + i,current_path +'/'+'dog')
            elif(i[:3] == "cat"):
                shutil.move(current_path +'/' + i,current_path +'/'+'cat')
      
    
  #create datasets
create_dataset("train_set",0.6)  
create_dataset("validation_set",0.5)
create_dataset("test_set",1)

sort_category("train_set")
sort_category("validation_set")
sort_category("test_set")



