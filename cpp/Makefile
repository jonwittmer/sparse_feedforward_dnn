CXX=g++
SOURCE=main.cpp sparse_layer.cpp sparse_model.cpp layer.cpp dense_layer.cpp
INCLUDE=-I${EIGEN_DIR} -I${JSON_DIR}
OUTPUT=test

default:
	${CXX} ${SOURCE} ${INCLUDE} -o ${OUTPUT} -march=native
