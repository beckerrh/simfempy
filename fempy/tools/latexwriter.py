# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import os
import numpy as np


#=================================================================#
class TableData(object):
    """
    n : first axis
    values : per method
    precs and types for latex
    """
    def __init__(self, n, values, type='float', prec=3, nname='n'):
        self.n = n
        self.values = values
        try:
            keys = list(values.keys())
        except:
            raise ValueError("values is not a dictionary (values=%s)" %values)
        self.precs = {}
        self.types = {}
        for key in keys:
            self.precs[key] = prec
            self.types[key] = type
        self.rotatenames = False
        self.nname = nname
    def computeDiffs(self):
        n, values, keys = self.n, self.values, list(self.values.keys())
        for key in keys:
            key2 = key + '-d'
            valorder = np.zeros(len(n))
            for i in range(1,len(n)):
                valorder[i] = abs(values[key][i]-values[key][i-1])
            values[key2] = valorder
            self.precs[key2] = self.precs[key]
            self.types[key2] = self.types[key]
    def computeReductionRate(self, diff=False):
        n, values, keys = self.n, self.values, list(self.values.keys())
        fi = 1+int(diff)
        for key in keys:
            if diff:
                if key[-2:] != "-d": continue
            key2 = key + '-o'
            valorder = np.zeros(len(n))
            for i in range(fi,len(n)):
                if not values[key][i-1]:
                    alpha = -1
                    continue
                try:
                    fnd = float(n[i])/float(n[i-1])
                    alpha = -2.0* np.log(values[key][i]/values[key][i-1]) / np.log(fnd)
                    # print 'n', n[i], n[i-1], values[key][i], values[key][i-1], " --> ", fnd, alpha
                except:
                    alpha=-1
                valorder[i] = alpha
            values[key2] = valorder
            self.precs[key2] = 2
            self.types[key2] = 'ffloat'

#=================================================================#
class LatexWriter(object):
    def __init__(self, dirname="Resultslatextest", filename=None):
        if filename is None:
            filename = dirname + ".tex"
        self.dirname = dirname + os.sep + "tex"
        if not os.path.isdir(self.dirname) :
            os.makedirs(self.dirname)
        self.latexfilename = os.path.join(self.dirname, filename)
        self.sep = '%' + 30*'='+'\n'
        self.data = {}
        self.countdata = 0
        print(self.dirname, self.latexfilename)

    def computeReductionRate(self, tabledata):
        n = tabledata.n
        values = tabledata.values
        keys = list(values.keys())
        orders = {}
        for key in keys:
            key2 = key + '-o'
            valorder = np.zeros(len(n))
            for i in range(1,len(n)):
                try:
                    fnd = float(n[i])/float(n[i-1])
                    alpha = -2.0* np.log(values[key][i]/values[key][i-1]) / np.log(fnd)
                    # print 'n', n[i], n[i-1], values[key][i], values[key][i-1], " --> ", fnd, alpha
                except:
                    alpha=-1
                valorder[i] = alpha
            values[key2] = valorder
            tabledata.precs[key2] = 2
            tabledata.types[key2] = 'ffloat'
            orders[key] = -1
            fnd = float(n[-1]) / float(n[0])
            if len(n)>1:
                try:
                    orders[key] = -2.0* np.log(values[key][-1]/values[key][0]) / np.log(fnd)
                except:
                    pass
        return tabledata, orders

    def append(self, n, values, type='float', name= None, redrate=False, diffandredrate=False):
        if name is None:
            name = 'table%1d' %self.countdata
        self.countdata += 1
        tabledata = TableData(n=n, values=values, type=type)
        if diffandredrate:
            tabledata.computeDiffs()
            tabledata.computeReductionRate(diff=True)
        if redrate:
            tabledata.computeReductionRate()
        self.data[name] = tabledata
    def write(self):
        self.latexfile = open(self.latexfilename, "w")
        self.writePreamble()
        for key,tabledata in sorted(self.data.items()):
            self.writeTable(name=key, tabledata=tabledata)
        self.writePostamble()
        self.latexfile.close()
    def __del__(self):
        try:
            self.latexfile.close()
        except:
            pass
    def writeTable(self, name, tabledata):
        n = tabledata.n
        values = tabledata.values
        nname = tabledata.nname
        keys_to_write = sorted(values.keys())
        size = len(keys_to_write)
        if size==0: return
        texta ='%\n%---\n%\n\\begin{table}[htp]\n\\begin{center}\n\\begin{tabular}{'
        texta += 'r|' + size*'|r' + '}\n'
        self.latexfile.write(texta)
        if tabledata.rotatenames:
            itemformated = "\sw{%s} &" %nname.replace('_','')
            for i in range(size-1):
                itemformated += "\sw{%s} &" %keys_to_write[i].replace('_','')
            itemformated += "\sw{%s}\\\\\\hline\hline\n" %keys_to_write[size-1].replace('_','')
        else:
            itemformated = "%15s " %nname.replace('_','')
            for i in range(size):
                itemformated += " & %15s " %keys_to_write[i].replace('_','')
            itemformated += "\\\\\\hline\hline\n"
        self.latexfile.write(itemformated)

        format_n = '%15d '
        formatvalue={}
        for i in range(size):
            key = keys_to_write[i]
            type = tabledata.types[key]
            prec = tabledata.precs[key]
            if  type == 'int':
                formatvalue[i] = ' & %15d '
            elif type=='float':
                formatvalue[i] = ' & %15.' + '%1de ' % prec
            elif type == 'ffloat':
                formatvalue[i] = ' & %15.' + '%1df ' % prec
            else:
                raise ValueError("no such type '%s'" %type)

        numberitems = len(n)
        for texline in range(numberitems):
            itemformated = format_n %(n[texline])
            for i in range(size):
                key = keys_to_write[i]
                itemformated += formatvalue[i] %values[key][texline]
            itemformated += "\\\\\\hline\n"
            self.latexfile.write(itemformated)
        texte='\\end{tabular}\n\\caption{%s}' %(name.replace('_','\_'))
        texte += "\n\\end{center}\n\\label{fig:ref}\n\\end{table}\n%\n%---\n%\n" 
        self.latexfile.write(texte)
    def writePreamble(self, name="none", rotatenames=False):
        texta = '\\documentclass[11pt]{article}\n\\usepackage[width=17cm, top=3mm, a4paper]{geometry}\n\\usepackage{times}\n\\usepackage{graphicx}\n\\usepackage{rotating}\n'
        # texta = texta+'\\topmargin -2cm\n\\oddsidemargin -2cm\n\\evensidemargin -2cm\n\\textwidth 17cm\n\\textheight 19cm\n '
        if rotatenames:
            texta += "\\newcommand{\sw}[1]{\\begin{sideways} #1 \\end{sideways}}\n"
        texta = texta + self.sep + '\\begin{document}\n' + self.sep + '\n'
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
        subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        command = "open " + filename.replace('.tex', '.pdf')
        subprocess.call(command, shell=True)


# ------------------------------------- #

if __name__ == '__main__':
    n = [i**2 for i in range(1, 5)]
    values={}
    values['u'] = np.random.rand((len(n)))
    values['v'] = np.random.rand((len(n)))
    latexwriter = LatexWriter()
    latexwriter.append(n, values)
    latexwriter.write()
    latexwriter.compile()
