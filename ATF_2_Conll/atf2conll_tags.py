import os
import click
from stat import ST_MODE, S_ISREG
import argparse
from shutil import rmtree
from converter import ATFCONLConvertor
from text2tag import TAGCLASS
#from pyoracc.atf.common.atffile import check_atf


def file_process(infile, output_path, taglist, verbose=False):
    outfolder = os.path.join(output_path, 'output_conll')
    print("\nouptput folder is {} \n".format(outfolder))
    if os.path.exists(outfolder):
    	rmtree(outfolder)
    if not os.path.exists(outfolder):
    	os.makedirs(outfolder)
#    try:
#        click.echo('Info: Checking {0} with Pyoracc atfchecker. \n'.format(infile))
#        check_atf(infile, 'cdli', verbose)
#    except SyntaxError as err:
#        click.echo('Error in file{0} with Pyoracc atfchecker.'.format(infile))
#        click.echo('Syntax error: {0}'.format(err))
        
    print("\n PROCESSING ATF FILES \n")
    
    convertor = ATFCONLConvertor(infile, output_path, taglist, verbose)
    convertor.convert()
    convertor.write2file()


def check_and_process(pathname, output_path, taglist, verbose=False):
    mode = os.stat(pathname)[ST_MODE]
    if S_ISREG(mode) and pathname.lower().endswith('.atf'):
        # It's a file, call the callback function
        if verbose:
            click.echo('Info: Processing {0}.'.format(pathname))
        file_process(pathname,output_path, taglist, verbose)



def main(input_path, output_path, verbose):
    POS_INPUT=output_path+'/pos_pipeline.txt'
    NER_INPUT=output_path+'/ner_pipeline.txt'    
    Obj=TAGCLASS(POS_INPUT,NER_INPUT)
    taglist=Obj.tag2list()
    
    if os.path.isdir(input_path):
        with click.progressbar(os.listdir(input_path), label='Info: Converting the files') as bar:
            for f in bar:
                pathname = os.path.join(input_path, f)
                check_and_process(pathname, output_path, taglist, verbose)
    else:
        check_and_process(input_path, output_path, taglist, verbose)

        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i","--input",help="Location of the atf file",default="ATF_INPUT/demo.atf")
    parser.add_argument("-o","--output",help="Location of the output folder for conll files",default="ATF_OUTPUT")
    parser.add_argument("-v","--verbose",help="printing details",default=False)
    
    args=parser.parse_args()
    #@click.command()
    #@click.option('--input_path', '-i', type=click.Path(exists=True, writable=True), prompt=True, required=True,
    #              help='Input the file/folder name.')
    #@click.option('-v', '--verbose', default=False, required=False, is_flag=True, help='Enables verbose mode')
    
    main(args.input, args.output, args.verbose)
    
    
    
    
    
    
        
