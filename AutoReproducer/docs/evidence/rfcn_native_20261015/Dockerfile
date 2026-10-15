FROM scratch
LABEL autorepro.native.owner="issue27-rfcn-native-20261015-05"
ADD ubuntu-base.tar.gz /
COPY ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
LABEL org.opencontainers.image.source="https://github.com/microsoft/caffe/tree/1a2be8ecf9ba318d516d79187845e90ac6e73197"
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list
RUN apt-get -o Acquire::Retries=2 -o Acquire::https::Timeout=30 update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends make gcc g++ cmake curl ca-certificates \
        libboost-system-dev libboost-thread-dev libboost-filesystem-dev libgflags-dev libgoogle-glog-dev \
        libhdf5-dev libleveldb-dev liblmdb-dev libopenblas-dev libsnappy-dev
RUN curl --fail --location --retry 2 --connect-timeout 20 --max-time 120 \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    --output /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && \
    apt-get -o Acquire::Retries=2 -o Acquire::https::Timeout=30 update && \
    apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 \
        libcublas-dev-12-8 libcurand-dev-12-8
ENV PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/caffe/build/lib:/opt/protobuf/lib:/usr/local/cuda/lib64
RUN curl --fail --location --retry 2 --connect-timeout 20 --max-time 180 \
    https://codeload.github.com/protocolbuffers/protobuf/tar.gz/refs/tags/v3.6.1 \
    --output /tmp/protobuf.tar.gz && \
    mkdir -p /opt/protobuf-source && tar -xzf /tmp/protobuf.tar.gz -C /opt/protobuf-source --strip-components=1 && \
    cmake -S /opt/protobuf-source/cmake -B /opt/protobuf-source/build \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/protobuf && \
    cmake --build /opt/protobuf-source/build -j4 && cmake --install /opt/protobuf-source/build
RUN curl --fail --location --retry 2 --connect-timeout 20 --max-time 180 \
    https://codeload.github.com/microsoft/caffe/tar.gz/1a2be8ecf9ba318d516d79187845e90ac6e73197 \
    --output /tmp/caffe.tar.gz && \
    mkdir -p /opt/caffe && tar -xzf /tmp/caffe.tar.gz -C /opt/caffe --strip-components=1 && \
    cmake -S /opt/caffe -B /opt/caffe/build \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed -lhdf5_serial_hl" \
        -DCPU_ONLY=OFF -DUSE_CUDNN=OFF -DUSE_OPENCV=OFF -DBUILD_python=OFF \
        -DBUILD_python_layer=OFF -DBUILD_matlab=OFF -DBUILD_docs=OFF -DBLAS=Open \
        -DCUDA_HOST_COMPILER=/usr/bin/gcc -DCUDA_ARCH_NAME=Manual \
        -DCUDA_ARCH_BIN=8.9 -DCUDA_ARCH_PTX=8.9 \
        -DCUDA_NVCC_FLAGS=--allow-unsupported-compiler \
        -DPROTOBUF_PROTOC_EXECUTABLE=/opt/protobuf/bin/protoc \
        -DPROTOBUF_INCLUDE_DIR=/opt/protobuf/include -DPROTOBUF_LIBRARY=/opt/protobuf/lib/libprotobuf.so && \
    cmake --build /opt/caffe/build --target caffe.bin -j4
CMD ["/opt/caffe/build/tools/caffe", "device_query", "--gpu=0"]
