import pathlib
import numpy as np
class Path(type(pathlib.Path())):
    def ls(self):
        return list(self.iterdir())
    def walk(self, inc_condition = None, exc_condition = None, max_depth=None):
        '''
        Find all the files contained within this folder recursively

        inc_condition (str): only include results containing this string
        exc_condition (str): don't include any results containing this string 
        max_depth (int): maximum depth of recursion, default None has no limit'''
        cont = True
        all_files = self.ls()
        depth = 1
        while cont==True:
            some_files = []
            for f in all_files:
                if f.is_dir():
                    [some_files.append(x) for x in f.ls()]
                else:
                    some_files.append(f)
            if depth == max_depth:
                cont = False
            else:
                cont = not np.all(np.asarray([x.is_file() for x in some_files]) == True)
                depth +=1
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
        if index != 0:
            return Path('/'.join(str(self).split('/')[:-index]) + f'/{end}')
        else:
            return Path(str(self)+f'/{end}')
    def mk(self):
        '''
        Make a directory for the path if one does not exist
        '''
        if not self.exists():
            self.mkdir()

def flatten_nav(sig):
    shape = [sig.shape[0]*sig.shape[1]]
    for i in sig.shape[2:]:
        shape.append(i)
    return sig.reshape(shape)
