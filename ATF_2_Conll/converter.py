import unicodedata
import codecs
import click
import os
import json

OUTPUT_FOLDER = 'output_conll'
with open('ATF_2_Conll/dictionary.json', "r") as read_file:
    SEGM_dict = json.load(read_file)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False






class ATFCONLConvertor:
    def __init__(self, inputFile, output_path, taglist, verbose):
        self.taglist=taglist
        self.inputFileName = inputFile
        self.outfolder = os.path.join(output_path, OUTPUT_FOLDER)
        self.verbose = verbose
        self.__reset__()

    def __reset__(self):
        self.outputFilename = ''
        self.surfaceMode = ''
        self.inEnvelope = ''
        self.column = ''
        self.tokens = []

    def convert(self):
        if self.verbose:
            click.echo('Info: Reading file {0}.'.format(self.inputFileName))
        with codecs.open(self.inputFileName, 'r', 'utf-8') as openedFile:
            for (i, line) in enumerate(openedFile):
                self.__parse(i, line.strip())

    def write2file(self):
        IDlist = list(map(lambda x: x[0], self.tokens))
        #if len(IDlist) != len(set(IDlist)):
        #    click.echo(
        #        'Error: File {0}, Text {1} : IDs generated are not unique'.format(self.inputFileName,
        #                                                                          self.outputFilename))
        outfile_name = os.path.join(self.outfolder, self.outputFilename + ".conll")
        
        print("Processing file {}".format(self.outputFilename))
        with codecs.open(outfile_name, 'w+', 'utf-8') as outputFile:
            outputFile.writelines("#new_text=" + self.outputFilename + "\n")
            outputFile.writelines("# ID\tFORM\tSEGM\tXPOSTAG\tHEAD\tDEPREL\tMISC\n")
            for tok in self.tokens:
                SEGM="_"
                tag="_"
                if (SEGM_dict.get(tok[1])!=None):
                    SEGM=SEGM_dict.get(tok[1])
                if(self.taglist.get(tok[1])!=None):
                    tag=self.taglist.get(tok[1])
                outputFile.writelines(tok[0] + '\t' + tok[1] + '\t' + SEGM + '\t' + tag + '\n')

    def __clean(self, tokenList):
        outTokenlist = []
        insert = True
        for tok in tokenList:
            if tok == u'($':
                insert = False
            elif tok == u'$)':
                insert = True
            elif insert:
                outTokenlist.append(tok)
            else:
                pass
        return outTokenlist

    def __parse(self, linenumber, line):
        tokenizedLine = line.split(" ")
        if len(line) == 0:
            pass
        elif line[0] == "&":
            if len(self.tokens) > 0:
                self.write2file()
            self.__reset__()
            firstword = tokenizedLine[0].lstrip("&")
            self.outputFilename = firstword
        elif line[0] == "@":
            # @(obverse[\?]?|reverse[\?]?|top[\?]?|bottom[\?]?|left[\?]?|right[\?]?|seal\s([A-Z]{1}|[0-9]+)?|surface [a-zA-Z0-9]+|face [a-zA-Z0-9]+)
            firstword = tokenizedLine[0].lstrip("@")
            if firstword == "obverse":
                self.surfaceMode = "o"
            elif firstword == "reverse":
                self.surfaceMode = "r"
            elif firstword == "top":
                self.surfaceMode = "t"
            elif firstword == "bottom" and len(tokenizedLine) == 1:
                self.surfaceMode = "b"
            elif firstword == "bottom":
                self.column = 'b' + tokenizedLine[-1]
            elif firstword == "left":
                self.surfaceMode = "l"
            elif firstword == "right":
                self.surfaceMode = "ri"
            elif firstword == "surface" or firstword == "face":
                self.surfaceMode = tokenizedLine[-1]
            elif firstword == "seal":
                self.surfaceMode = "s" + tokenizedLine[-1]
                self.inEnvelope = ''
            elif firstword == "envelope":
                self.inEnvelope = 'e'
            elif firstword == "column":
                self.column = 'col' + tokenizedLine[-1]
            elif firstword == 'tablet' or firstword == 'object':
                if self.verbose:
                    pass
                    # click.echo('File {0}, Linenumber {1} : Found a tablet or object in {2}'.format(self.inputFileName,linenumber, line))
            else:
                if self.verbose:
                    click.echo(
                        'Warning: File {0}, Linenumber {1} : Unrecognized @ in {2}'.format(self.inputFileName,
                                                                                           linenumber, line))
        elif is_number(line[0]):
            linenumber = tokenizedLine[0].rstrip(".")
            tokensToProcess = tokenizedLine[1:]
            cleanTokensToProcess = self.__clean(tokensToProcess)
            for i in range(len(cleanTokensToProcess)):
                prefix = self.inEnvelope + self.surfaceMode
                if self.column == '':
                    IDlist = [prefix, linenumber, str(i + 1)]
                else:
                    IDlist = [prefix, self.column, linenumber, str(i + 1)]
                ID = ".".join(IDlist)
                form = cleanTokensToProcess[i]
                form_clean = form.replace('#', '').replace('[', '').replace(']', '').replace('<', '').replace('>',
                                                                                                              '').replace(
                    '!', '').replace('?', '').replace('@c','').replace('@t','').replace('_','').replace(',','')    
                self.tokens.append((ID, form_clean))
            
                
                
                
                
                
                
                
                
                
                
                
                
                
