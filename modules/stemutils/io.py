import pathlib
import numpy as np
class Path(type(pathlib.Path())):
    def ls(self):
        return list(self.iterdir())
    def walk(self, inc_condition = None, exc_condition = None):
        '''
        Find all the files contained within this folder recursively

        inc_condition (str): only include results containing this string
        exc_condition (str): don't include any results containing this string '''
        cont = True
        all_files = self.ls()
        while cont==True:
            some_files = []
            for f in all_files:
                if f.is_dir():
                    [some_files.append(x) for x in f.ls()]
                else:
                    some_files.append(f)
            cont = not np.all(np.asarray([x.is_file() for x in some_files]) == True)
            all_files = some_files.copy()
        if inc_condition != None:
            all_files = [x for x in all_files if str(x).find(inc_condition) != -1]
        if exc_condition != None:
            all_files = [x for x in all_files if str(x).find(exc_condition) == -1]
        return all_files
    def redirect(self, end, index = 1):
        '''
        Replace the end target of the Path spliting at the i-th / from the end
        
        end (str): new target to append to path
        index (int): position to append new target
        
        eg. Path('a/b/c/d/e').redirect('f/g', 2) --> Path('a/b/c/f/g') '''
        return Path('/'.join(str(self).split('/')[:-index]) + f'/{end}')

def flatten_nav(sig):
    shape = [sig.shape[0]*sig.shape[1]]
    for i in sig.shape[2:]:
        shape.append(i)
    return sig.reshape(shape)
