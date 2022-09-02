CXX=g++
SOURCE=main.cpp sparse_layer.cpp sparse_model.cpp
INCLUDE=-I${EIGEN_DIR} -I/workspace/jonathan/Programs/json/single_include
OUTPUT=test

default:
	${CXX} ${SOURCE} ${INCLUDE} -o ${OUTPUT} -march=native
