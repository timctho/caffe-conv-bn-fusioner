# Caffe Tool to fuse Conv-BN-Scale


```
usage: run.py [-h] [-d DEPLOY] [-m MODEL] [-t] [--height HEIGHT]
              [--width WIDTH]

  -d DEPLOY, --deploy DEPLOY
                        deploy prototxt
  -m MODEL, --model MODEL
                        caffemodel
  -t, --test            run test
  --height HEIGHT       Used to generate random sample if need to run test
  --width WIDTH         Used to generate random sample if need to run test
```

New *.prototxt* and *.caffemodel* will be saved in "_folded" postfix.