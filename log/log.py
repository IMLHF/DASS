
class LOG(object):
  def __init__(self,file):
    self.file=file

  def print_save(self,msg):
    print(msg)
    with open(self.file,'a+') as f:
      f.writelines(str(msg)+'\n')

  def save(self,msg):
    with open(self.file,'a+') as f:
      f.writelines(str(msg)+'\n')
