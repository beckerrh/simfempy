# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import os, platform
import numpy as np


#=================================================================#
class TableData(object):
    """
    n : first axis
    values : per method
    """
    def __init__(self, **kwargs):
        values = kwargs.pop('values')
        if not isinstance(values, dict):
            raise ValueError("values is not a dictionary (values=%s)" %values)
        self.values = dict((str(k), v) for k, v in values.items())
        self.n = kwargs.pop('n')
        for k,v in values.items():
            if len(self.n) != len(v):
                raise ValueError(f"wrong lengths: n({len(self.n)}) value({len(v)})\n{values=}")
        self.nname = kwargs.pop('nname')
        self.nformat = "{:15d}"
        if 'nformat' in kwargs:
            self.nformat = "{{:{}}}".format(kwargs.pop('nformat'))
        self.valformat = {k:"{:10.2e}" for k in values.keys()}
        if 'valformat' in kwargs:
            valformat = kwargs.pop('valformat')
            if isinstance(valformat,str):
                for k in values.keys():
                    self.valformat[k] = "{{:{}}}".format(valformat)
            else:
                for k in values.keys():
                    self.valformat[k] = "{{:{}}}".format(valformat[k])
    def computePercentage(self):
        self.values = dict((k+"(\%)", v) for k, v in self.values.items())
        self.values['sum'] = np.zeros(len(self.n))
        for i in range(len(self.n)):
            sum = 0
            for key, value in self.values.items():
                sum += self.values[key][i]
            self.values['sum'][i] = sum
            for key, value in self.values.items():
                if key=='sum': continue
                self.values[key][i] *= 100/sum
        for key in self.values.keys():
            self.valformat[key] = "{:8.2f}"
        self.valformat['sum'] = "{:10.2e}"
    def computeDiffs(self):
        n, values, keys = self.n, self.values, list(self.values.keys())
        for key in keys:
            key2 = key + '-d'
            valorder = np.zeros(len(n))
            for i in range(1,len(n)):
                valorder[i] = abs(values[key][i]-values[key][i-1])
            values[key2] = valorder
            self.valformat[key2] = self.valformat[key]
            self.valformat[key2] = self.valformat[key]
    def computeReductionRate(self, dim, diff=False):
        n, values, keys = self.n, self.values, list(self.values.keys())
        if not isinstance(n[0],(int,float)): raise ValueError("n must be int or float")
        fi = 1+int(diff)
        for key in keys:
            if diff:
                if key[-2:] != "-d": continue
            key2 = key + '-o'
            valorder = np.zeros(len(n))
            for i in range(fi,len(n)):
                if not values[key][i-1]:
                    p = -1
                    continue
                fnd = float(n[i])/float(n[i-1])
                vnd = values[key][i]/values[key][i-1]
                if abs(vnd)>1e-10:
                    p = -dim* np.log(vnd) / np.log(fnd)
                else:
                    p=-1
                valorder[i] = p
            values[key2] = valorder
            self.valformat[key2] = "{:8.2f}"

#=================================================================#
class LatexWriter(object):
    # def __init__(self, dirname="Resultslatextest", filename=None):
    def __init__(self, **kwargs):
        self.dirname = "Resultslatextest"
        if "dirname" in kwargs: self.dirname = kwargs.pop("dirname")
        filename = self.dirname + ".tex"
        if "filename" in kwargs: filename = kwargs.pop("filename")
        if filename[-4:] != '.tex': filename += '.tex'
        self.dirname += os.sep + "tex"
        if not os.path.isdir(self.dirname): os.makedirs(self.dirname)
        self.latexfilename = os.path.join(self.dirname, filename)
        self.author, self.title = self.__class__.__name__, "No title given"
        if "title" in kwargs: self.title = kwargs.pop("title")
        if "author" in kwargs: self.author = kwargs.pop("author")
        self.sep = '%' + 30*'='+'\n'
        self.data = {}
        self.countdata = 0
        self.rotatenames = True
    def __del__(self):
        try:
            self.latexfile.close()
        except:
            pass
    def append(self, **kwargs):
        if 'name' in kwargs: name = kwargs.pop('name')
        else: name = 'table_{:d}'.format(self.countdata+1)
        self.countdata += 1
        tabledata = TableData(**kwargs)
        if 'diffandredrate' in kwargs and kwargs.pop('diffandredrate'):
            tabledata.computeDiffs()
            tabledata.computeReductionRate(kwargs.pop('dim'), diff=True)
        if 'redrate' in kwargs and kwargs.pop('redrate'):
            tabledata.computeReductionRate(kwargs.pop('dim'))
        if 'percentage' in kwargs and kwargs.pop('percentage'):
            tabledata.computePercentage()
        self.data[name] = tabledata
    def write(self, sort=False):
        self.latexfile = open(self.latexfilename, "w")
        if len(self.data) < 4: self.rotatenames=False
        self.writePreamble()
        if sorted:
            for key,tabledata in sorted(self.data.items()):
                self.writeTable(name=key, tabledata=tabledata)
        else:
            for key, tabledata in self.data.items():
                self.writeTable(name=key, tabledata=tabledata)
        self.writePostamble()
        self.latexfile.close()
    def writeTable(self, name, tabledata):
        n, nname, nformat, values, valformat = tabledata.n, tabledata.nname, tabledata.nformat, tabledata.values, tabledata.valformat
        nname = nname.replace('_', '')
        name = name.replace('_', '')
        keys_to_write = sorted(values.keys())
        size = len(keys_to_write)
        if size==0: return
        texta ='\\begin{table}[!htbp]\n\\begin{center}\n\\begin{tabular}{'
        texta += 'r|' + size*'|r' + '}\n'
        self.latexfile.write(texta)
        if self.rotatenames:
            itemformated = "\sw{%s} &" %nname
            for i in range(size-1):
                itemformated += "\sw{%s} &" %keys_to_write[i].replace('_','')
            itemformated += "\sw{%s}\\\\\\hline\hline\n" %keys_to_write[size-1].replace('_','')
        else:
            itemformated = "%15s " %nname
            for i in range(size):
                itemformated += " & %15s " %keys_to_write[i].replace('_','')
            itemformated += "\\\\\\hline\hline\n"
        self.latexfile.write(itemformated)
        for texline in range(len(n)):
            itemformated = nformat.format(n[texline])
            for i in range(size):
                key = keys_to_write[i]
                itemformated += '&'+valformat[key].format(values[key][texline])
            itemformated += "\\\\\\hline\n"
            self.latexfile.write(itemformated)
        # texte = '\\end{tabular}\n\\caption{%s}' %(name)
        texte = f"\\end{{tabular}}\n\\caption{{{name}}}"
        texte += f"\n\\end{{center}}\n\\label{{tab:{name}}}\n\\end{{table}}\n"
        texte += f"%\n%{30*'-'}\n%\n"
        self.latexfile.write(texte)
    def writePreamble(self, name="none"):
        texta = '\\documentclass[11pt]{article}\n\\usepackage[margin=3mm, a4paper]{geometry}\n'
        texta += '\\usepackage{times,graphicx,rotating,subfig}\n'
        if self.rotatenames:
            texta += "\\newcommand{\sw}[1]{\\begin{sideways} #1 \\end{sideways}}\n"
        texta += f"\\author{{{self.author}}}\n"
        texta += f"\\title{{{self.title}}}\n"
        texta += self.sep + '\\begin{document}\n' + self.sep + '\n'
        texta += "\\maketitle\n"
        texta += f"%\n%{30*'-'}\n%\n"
        self.latexfile.write(texta)
    def writePostamble(self):
        texte = '\n' + self.sep + '\\end{document}\n' + self.sep
        self.latexfile.write(texte)
        self.latexfile.close()
    def compile(self):
        import subprocess
        os.chdir(self.dirname)
        filename = os.path.basename(self.latexfilename)
        command = "pdflatex " + filename
        try:
            result = subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result: raise ValueError("command pdflatex not found")
        except:
            raise ValueError("no pdflatex found") 
            pass
        if platform.system() == "Linux":
            command = "xdg-open "
        elif platform.system() == "Darwin":
            command = "open "
        elif platform.system() == "Windows":
            command = "explorer "
        command += filename.replace('.tex', '.pdf')
        try: 
            subprocess.call(command, shell=True)
        except: 
            pass


# ------------------------------------- #
if __name__ == '__main__':
    n = [i**2 for i in range(1, 10)]
    values={}
    values['u'] = np.power(n,-2.) + 0.01*np.random.rand((len(n)))
    values['v'] = np.power(n,-3.) + 0.01*np.random.rand((len(n)))
    latexwriter = LatexWriter()
    latexwriter.append(n=n, nname='n', values=values, diffandredrate=True, dim=2)
    values2={}
    values2[1] = [1,2,3]
    values2[2] = [4,5,6]
    latexwriter.append(n=['a','b','c'], nname='letter', nformat="10s", values=values2, percentage=True)
    values3={}
    values3['1'] = np.linspace(1,3,5)
    latexwriter.append(n=np.arange(5), nname= 'toto', nformat="4d", values=values3)
    latexwriter.write()
    latexwriter.compile()
