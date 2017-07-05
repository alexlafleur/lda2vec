lda2vec: Tools for interpreting natural language
=================================================

Dockerfile with prerequsites:
_____________________________

::

  FROM nvidia/cuda

  RUN	apt update && apt-get -y upgrade
  RUN 	apt-get -y install python-pip
  RUN	apt-get -y install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev
  RUN	pip install -U setuptools
  RUN 	pip install chainer --no-cache-dir -vvvv
  RUN	pip install numpy scipy pandas pyxdameraulevenshtein sklearn spacy h5py cupy gensim
  
  ENTRYPOINT /bin/bash


To Do
____________________________

- try lda2vec with german Word2Vec Vector:

  - pre-built: https://tubcloud.tu-berlin.de/public.php?service=files&t=dc4f9d207bcaf4d4fae99ab3fbb1af16
  - build our own using https://github.com/devmount/GermanWordEmbeddings
  - How to handle Emoticons? See here:
      - https://apps.timwhitlock.info/emoji/tables/unicode#block-1-emoticons
      - https://github.com/laurenancona/twimoji/blob/gh-pages/twitterEmojiProject/emoticon_conversion_noGraphic.csv CSV File containing Unicode Emoticons with Description
