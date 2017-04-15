Authors: Cuiqing Li,Bo Liu, Yunhan Zhao

Location: Department of Computer Science of the Johns Hopkins University

### Description:
We implemented an extended LDA models(Topic Model) which is used to do Document classification. 
For more details, please check PDF file. This Project is from our machine learning with Data to Model class. 

To run our collaspsed-sampler.py file, please type the following command:
```
python2 collapsed-sampler.py input-train.txt input-test.txt output.txt 10 0.5 0.1 0.01 1100 1000
```

Also, if you want to make the training become fast, please change the command lines into:

```
pypy collapsed-sampler.py input-train.txt input-test.txt output.txt 10 0.5 0.1 0.01 1100 1000
```
In other words, you have to install [pypy](http://pypy.org/download.html) to make the trainning become fast, or you can install Cython to make the training become fast! More details are displayed in the PDF file. Typically, as for every iteration, using pypy only takes 0.5s roughly. 



