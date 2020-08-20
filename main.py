#!/usr/bin/python
import os
import zipfile, io, pathlib
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename

HOSTNAME = '0.0.0.0'
PORT = '8080'
UPLOAD_FOLDER = './ATF_INPUT/'
OUTPUT_FOLDER='./ATF_OUTPUT/'
ALLOWED_EXTENSIONS = {'atf','txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def do_translation(path):
    ATF="True"
    if (path.split(".")[-1]=='txt'):
    	ATF="False"
    os.system(f'python3 pipeline.py -i {path} -a {ATF}')
    return "This is a translation."

@app.route('/translate/<filename>')
def translate(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #with open(path) as fp:
    #file_contents = fp.read()
    try:
        translation = do_translation(path)  
    except Exception as error:
        app.logger.error("File format is not supported file", error)
    return redirect('/downloadfiles')
    

@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file in post data')
            return redirect( request.url )
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No file selected')
            return redirect( request.url )
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('translate',
                                    filename=filename))
        else:
            flash('Filetype not allowed')
            return redirect( request.url )
    return render_template("upload.html")


# Download API
@app.route("/downloadfiles", methods = ['GET'])
def download_file():
    return render_template('download.html')
    
    
@app.route('/returnfiles', methods = ['GET'])
def return_files():
    base_path = pathlib.Path('./ATF_OUTPUT')
    data = io.BytesIO()
    with zipfile.ZipFile(data, mode='w') as z:
        for f_name in base_path.iterdir():
            z.write(f_name)
    data.seek(0)
    return send_file(
        data,
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename='results.zip',
        cache_timeout=0
    )

@app.route('/returnfilesconll', methods = ['GET'])
def return_files_conll():
    base_path = pathlib.Path('./ATF_OUTPUT/output_conll')
    data = io.BytesIO()

    try:
        with zipfile.ZipFile(data, mode='w') as z:
            for f_name in base_path.iterdir():
                z.write(f_name)
        data.seek(0)
    except Exception as error:
        app.logger.error("conll-u is genrated only for atf files")
        return render_template('download.html')

    return send_file(
        data,
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename='conll-u_results.zip',
        cache_timeout=0
    )

#    file_path = OUTPUT_FOLDER + filename
#    return send_file(file_path, as_attachment=True)




if __name__ == '__main__':
    app.run(
            debug=True, 
            host=HOSTNAME,
            port=PORT,
        )
