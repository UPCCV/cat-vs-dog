import os,argparse,random,shutil
from random import shuffle

def get_files_by_category(args,set):
    datadir=args.datadir#+"/"+set
    print("loading data from "+datadir+":")
    file=open("util/"+set+".txt","w");
    categoryfile=open("modeldef/labels.txt",'w')
    subdirs=os.listdir(datadir)
    classindex=0
    paths=[]
    categorys=[]
    for subdir in subdirs:
        if(os.path.isdir(datadir+"/"+subdir)):
            categorys.append(str(classindex)+" "+subdir)
            files=os.listdir(datadir+"/"+subdir)
            files=[file for file in files]
            random.shuffle(files)
            print(subdir,len(files))
            for f in files:
                paths.append(subdir+"/"+f+" "+str(classindex)+"\n")
            classindex=classindex+1
    for category in categorys:
        if category==categorys[1]:
            categoryfile.write(category)
        else:
            categoryfile.write(category+"\n")
    random.shuffle(paths)
    print("writing to "+set+".txt")
    for path in paths:
        file.write(path)
    print(len(paths))

def get_datasets(args):
    sets=["train","val"]
    for set in sets:
        get_files_by_category(args,set)
    print("generate_txt4lmdb Done")

def get_files(args):
    datadir=args.datadir
    files=os.listdir(datadir)
    lines=[]
    for file in files:
        items=file.split(".")
        path=file#"data/train/"+
        if items[0]=="cat":
            line=path+" 0"
        else:
            line=path+" 1"
        lines.append(line)
    shuffle(lines)
    with open("util/train.txt","w") as ftrain:
        with open("util/val.txt","w") as fval:
            for i in range(len(lines)):
                line=lines[i]
                if i<0.8*len(lines):
                    ftrain.write(line+"\n")
                else:
                    fval.write(line+"\n")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",default="../data/train",help="Directory of images to classify")
    #parser.add_argument("--set",default="train",help="set")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args=get_args()
    #get_datasets(args)
    get_files(args)