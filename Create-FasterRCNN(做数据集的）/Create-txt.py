import os
import random



# fname=input('Enter filename:')
# try:
#     fobj=open(fname,'a')                 # 这里的a意思是追加，这样在加了之后就不会覆盖掉源文件中的内容，如果是w则会覆盖。
# except IOError:
#     print('*** file open error:')
# else:
#     data_dir='png/'
#     for file in os.listdir(data_dir):
#         name = file.split('.')
#         fobj.write(str(name[0])+'.'+str(name[1])+'\n')   #  这里的\n的意思是在源文件末尾换行，即新加内容另起一行插入。
#     fobj.close()                              #   特别注意文件操作完毕后要close
# input('Press Enter to close')






data_dir='rgbd_dataset2/'
classes={'apple','ball','banana','bowl','garlic','greens','mushroom','lemon','onion','orange','peach','pear','potato','tomato'}
fname=input('Enter filename:')
name_=fname.split('.')
name=name_[0].split('_')
try:
    fobj=open(fname,'a')
except IOError:
    print('*** file open error:')
else:
    for NAME in classes:
        path = data_dir+ NAME + '/'
        for file in os.listdir(path):
            sname = file.split('.')
            if  name[0]==NAME:
                label=1
            else:
                label=-1
            if label==1:
               fobj.write(str(sname[0])+'\n')
            else:
               fobj.write(str(sname[0])+'\n')
    fobj.close()                              #   特别注意文件操作完毕后要close
input('Press Enter to close')
