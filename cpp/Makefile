CXX=icpc
#SOURCE=main.cpp sparse/sparse_layer.cpp sparse/sparse_model.cpp sparse/layer.cpp sparse/dense_layer.cpp autoencoder/compressed_batch.cpp autoencoder/autoencoder.cpp
INCLUDES := -I${EIGEN_DIR} -I${JSON_DIR} -I./ #-Isparse/ -Inormalization/ -Iautoencoder/ -Iutils/
BUILD_DIR := ./build
OBJS := $(BUILD_DIR)/layer.o $(BUILD_DIR)/sparse_layer.o $(BUILD_DIR)/dense_layer.o  $(BUILD_DIR)/sparse_model.o $(BUILD_DIR)/compressed_batch.o $(BUILD_DIR)/autoencoder.o $(BUILD_DIR)/compression_base.o $(BUILD_DIR)/batch_preparer.o  $(BUILD_DIR)/autoencoder_interface_c.o  ${BUILD_DIR}/autoencoder_debug.o
CXX_FLAGS := -xCORE-AVX512 -qopt-zmm-usage=high -O3 -g -mkl

all: $(BUILD_DIR)/libautoencoder_c.so test

$(BUILD_DIR)/layer.o: sparse/layer.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -march=native ${CXX_FLAGS}

$(BUILD_DIR)/sparse_layer.o: sparse/sparse_layer.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -march=native ${CXX_FLAGS} 

$(BUILD_DIR)/sparse_model.o: sparse/sparse_model.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@ -march=native ${CXX_FLAGS} 

$(BUILD_DIR)/dense_layer.o: sparse/dense_layer.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native ${CXX_FLAGS}

$(BUILD_DIR)/compressed_batch.o: batch_preparation/compressed_batch.cpp
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native ${CXX_FLAGS}

$(BUILD_DIR)/batch_preparer.o: batch_preparation/batch_preparer.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native ${CXX_FLAGS} 

$(BUILD_DIR)/compression_base.o: autoencoder/compression_base.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native ${CXX_FLAGS} 

$(BUILD_DIR)/autoencoder.o: autoencoder/autoencoder.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native ${CXX_FLAGS} 

$(BUILD_DIR)/autoencoder_debug.o: autoencoder/autoencoder_debug.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native ${CXX_FLAGS} 

$(BUILD_DIR)/autoencoder_interface_c.o: autoencoder/autoencoder_interface_c.cpp 
	$(CXX) ${INCLUDES}  -c -fPIC $^ -o $@  -march=native ${CXX_FLAGS} 

$(BUILD_DIR)/libautoencoder_c.so: $(OBJS)
	$(CXX) ${INCLUDES}  -shared -o $@ $^ -march=native ${CXX_FLAGS} -lpthread 

test: main.cpp $(BUILD_DIR)/libautoencoder_c.so
	mpicc ${INCLUDES} $^ -o $@ -march=native ${CXX_FLAGS} -L./build -lautoencoder_c -qopenmp
