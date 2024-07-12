# Containerizing Microsoft's Kosmos-2.5 Multimodal-LLM (MLLM) for Local OCR via a RESTful API

[Official Kosmos-2.5 Repo](https://github.com/microsoft/unilm/tree/master/kosmos-2.5)

[Official Flash Attention Repo](https://github.com/Dao-AILab/flash-attention)


| ![Image 1](kosmos-2_5-container-files/kosmos-2_5/assets/example/in.png) | ![Image 2](kosmos-2_5-container-files/kosmos-2_5/assets/example/in.png) | ![Image 3](kosmos-2_5-container-files/kosmos-2_5/assets/example/in.png) |
|:-----------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:-----------------------------------------------------------------------:|
|                           **(a) Input**                                 |                        **(b) Using the ocr prompt**                     |                        **(c) Using the markdown prompt**                |


## Introduction

### The Significance of OCR in the LLM-Landscape

Optical Character Recognition, commonly abbreviated as "OCR", is a technology to recognize and extract text from visual data such as images. OCR is often essential for a myriad of applications where simpler techniques fall short or fail entirely.

One such application is the extraction of text from documents, such as PDFs. Oftentimes, PDFs comprise of documents obtained from a scanner wherein each page of the generated PDF is basically an image, and thus simply attempting to read & parse text out of the document will not work, as no text exists! Further, even in cases where text can be parsed from PDFs, the complex and varied internal structure of PDF documents often results in the extracted text being garbled (mis-spelled or incorrectly split/combined) and lacking formatting integrity of the source document. For any documents containing crucial data, these issues can be serious as the essence of the data could be lost or corrupted via such extracts. This challenge is prevalent in document formats other than PDFs too, and high-performance OCR tools are often the very best in addressing these challenges.

While document-centric text-extraction has always been a popular requirement, this usecase is especially in the limelight today with the surge in popularity of Large Language Models (LLMs) and specifically, their use in RAG (Retrieval Augmented Generation) applications, wherein users upload their own documents and engage in conversations wherein LLMs ground their responses in the uploaded content, all in an effort to mitigate the pitfalls of AI-generated inaccuracies or "hallucinations." [LARS - The LLM & Advanced Referencing Solution](https://github.com/abgulati/LARS), is one such application which additionally injects detailed citations into responses to further increase trust in LLM outputs.

### RAG-Refresher

The typical pipeline of such RAG-applications involves extracting textual data from uploaded documents, breaking up this extract into fixed-sized chunks for processing via an embedding model and finally, storing the resulting embeddings into a vector database. Subsequently on receiving a user query, a semantic-similarity search is carried out on the vector database, relevant chunks of data retrieved, and a large corpus of contextual data supplied to the LLM alongside the user's query, all of which (hopefully!) results in significantly higher quality response generation by the LLM. Thus the term 'Retrieval Augmented Generation'!

It's easy to focus on optimizing such RAG-pipelines by focusing on the LLMs & embedding models used, and on the chunking strategy (size, overlap, etc) applied. However, the very first step of text-extraction itself, if not done with a high degree of precision, risks compromising all downstream tasks even if State-Of-The-Art (SOTA) models & techniques are deployed! It's thus very much worthwhile to expend the necessary resources to optimize this first step of the RAG pipeline adequately.

In doing so, it's soon discovered that local OCR techniques based on popular tools like Tesseract or Camelot often require extensive tuning to the input image and thus fall short in RAG applications that aim to be as broadly applicable as possible, allowing the user to upload a variety of document formats, types & content. Indeed, commercial OCR services served up by cloud providers such as Azure are often a necessity, and applications might incorporate OCR via API calls as a result. This is also the approach that's adopted in LARS, wherein an [extensive investigation of local OCR tools and transformer models clearly indicated the necessity of such cloud services](https://github.com/abgulati/LARS/blob/main/documents/refinements_research/Improving%20Text%20Extraction%20-%20Feb2024.pptx).

However, with the advent of a new breed of Multimodal-LLMs (MLLMs), and more specifically vision-LLMs, the landscape for local OCR may change significantly, with high-quality text-identification & extraction provided by locally run models that natively process visual data.

### Kosmos-2.5

Microsoft's Kosmos-2.5 is one such MLLM and in Microsoft's own words, is specifically a ["literate model for machine reading of text-intensive images."](https://github.com/microsoft/unilm/tree/master/kosmos-2.5#kosmos-25-a-multimodal-literate-model)

However, it's no easy feat to get this model up and running locally on your device: with a very stringent and specific set of hardware and software requirements, this model is extremely temperamental to deploy and use! Popular backends such as [llama.cpp don't support it](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#description) and a [very specific, non-standard and customized version of the transformers library](https://github.com/Dod-o/kosmos2.5_tools/tree/transformers) is required to correctly infer it. Certain [dependencies necessitate Linux](https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility), while others necessitate [very specific generations of Nvidia GPUs](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).

With such specific dependencies that can even hinder the deployment of other local LLMs, how can this model be made to co-exist alongside actual real-world applications and more crucially, be made to serve those applications in a useful manner?

While I cannot liberate the hardware requirements, I did see an opportunity to address the software challenges: by containerizing the model and its dependencies and leveraging PyFlask to expose the model over a RESTful API, Kosmos-2.5 can be made available as a service, thus providing fully local & high-performance OCR capabilities by leveraging a cutting-edge MLLM!


## Table of Contents

1. [Containerizing Microsoft's Kosmos-2.5 Multimodal-LLM (MLLM) for Local OCR via a RESTful API](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#containerizing-microsofts-kosmos-25-multimodal-llm-mllm-for-local-ocr-via-a-restful-api)
2. [Introduction](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#introduction)
    - [The Significance of OCR in the LLM-Landscape](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#the-significance-of-ocr-in-the-llm-landscape)
    - [RAG-Refresher](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#rag-refresher)
	- [Kosmos-2.5](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#kosmos-25)
2. [Dependencies](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#dependencies)
    - [1. Nvidia Ampere, Hopper or Ada-Lovelace GPU with minimum 12GB VRAM](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#1-nvidia-ampere-hopper-or-ada-lovelace-gpu-with-minimum-12gb-vram)
    - [2. Nvidia CUDA v12.4.1](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#2-nvidia-cuda-v1241)
    - [3. Docker (with WSL2 on Windows11)](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#3-docker-with-wsl2-on-windows11)
    - [4. Nvidia Container Toolkit](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#4-nvidia-container-toolkit)
3. [Installing & Deploying the Kosmos-2.5 Pre-Built Docker Image](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#installing--deploying-the-kosmos-25-pre-built-docker-image)
4. [Building the Docker Image](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#building-the-docker-image)
5. [API Specification](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#api-specification)
6. [Invoke Kosmos-2.5 API - /infer endpoint](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#invoke-kosmos-25-api---infer-endpoint)
    - [via POSTMAN](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#via-postman)
    - [via CURL](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#via-curl)
    - [via Python Requests](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#via-python-requests)
    - [via JavaScript - Fetch](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#via-javascript---fetch)
    - [via JavaScript - jQuery](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#via-javascript---jquery)
7. [Rebuilding the Dependencies & Container - If the Pre-Built Image & dockerfile in this Repo Fail to Work](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#rebuilding-the-dependencies--container---if-the-pre-built-image--dockerfile-in-this-repo-fail-to-work)
    - [Option 1 (recommended) - Pre-Build Dependencies in Host Machine & Re-use for `docker build`](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#option-1-recommended---pre-build-dependencies-in-host-machine--re-use-for-docker-build)
    - [Option 2 (very slow) - Build Dependencies Within Container with `docker build`](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#option-2-very-slow---build-dependencies-within-container-with-docker-build)
8. [Running Kosmos-2.5 Uncontainerized](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#running-kosmos-25-uncontainerized)


## Dependencies

### 1. Nvidia Ampere, Hopper or Ada-Lovelace GPU with minimum 12GB VRAM

- Due to the use & requirements of [Flash Attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features), you must have an Nvidia GPU based on one of the below architecture families:

    - Ampere: RTX 3000 GeForce, RTX A Professional series of GPUs, A100 etc.
    - Hopper: H100, H200, and H800
    - Ada-Lovelace: RTX 4000 GeForce or Professional series of GPUs

- The model consumes 10GB of VRAM in my testing, which further limits it to the below GPUs:

    - RTX 3060 or RTX 3080 (might work on 10GB variant, 12GB variant preferrable) RTX 3080 Ti, RTX 3090, RTX 3090 Ti and Laptop RTX 3080 (16GB VRAM variant) & RTX 3080 Ti GPUs
    - RTX A800, A4000 and above Professional GPUs
    - A100, H100, H200, and H800
    - RTX 4070 Ti Super, RTX 4080, RTX 4080 Super, RTX 4090 and Laptop RTX 4080 and RTX 4090 GPUs
    - RTX 2000 & above Ada Lovelace Professional GPUs

### 2. Nvidia CUDA v12.4.1

- Install Nvidia [GPU Drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)

- Install Nvidia [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) - Kosmos-2.5 container built with v12.4.1

- Verify Installation via the terminal:
    ```
    nvcc -V
    nvidia-smi
    ```

- If you encounter `nvcc not found` errors on Linux, you must manually set the NVCC PATH:

    - confirm symlink for cuda:

        ```
        ls -l /usr/local/cuda
        ls -l /etc/alternatives/cuda
        ```

    - update bashrc:

        ```
        nano ~/.bashrc

        # add this line to the end of bashrc:
        export PATH=/usr/local/cuda/bin:$PATH
        ```

    - reload bashrc:

        ```
        source ~/.bashrc
        ``` 

    - Confirm nvcc is correctly setup:

        ```
        nvcc -V
        ``` 

### 3. Docker (with WSL2 on Windows11)

- While not explicitly required, some experience with Docker containers and familiarity with the concepts of containerization and virtualization is recommended!

1. Installing Docker

    - Your CPU should support virtualization and it should be enabled in your system's BIOS/UEFI

    - Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

    - If on Windows, you may need to install the Windows Subsystem for Linux if it's not already present. To do so, open PowerShell as an Administrator and run the following:

        ```
        wsl --install
        ```

    - Ensure you have WSL version 2 by running:

        ```
        wsl -v
        # or
        wsl --status
        ```

        Update WSL if not!

    - Ensure Docker Desktop is up and running, then open a Command Prompt / Terminal and execute the following command to ensure Docker is correctly installed and up and running:

        ```
        docker ps
        ```

2. Windows Only - Install Ubuntu 22.04 via the Microsoft Store if it's not already installed:

    - Open the Microsoft Store app on your PC, and download & install Ubuntu 22.04.3 LTS
    
        <p align="center">
        <img src="https://github.com/abgulati/kosmos-2_5-containerized/blob/main/images/ubuntu_for_docker_wsl.png"  align="center">
        </p>

    - Launch an Ubuntu shell in Windows by searching for ```Ubuntu``` in the Start-menu after the installation above is completed

3. Windows Only - Docker & WSL Integration:

    - Open a new PowerShell window and set this Ubuntu installation as the WSL default:

        ```
        wsl --list
        wsl --set-default Ubuntu-22.04 # if not already marked as Default
        ```

    -  Navigate to ```Docker Desktop -> Settings -> Resources -> WSL Integration``` -> Check Default & Ubuntu 22.04 integrations. Refer to the screenshot below:
    
        <p align="center">
        <img src="https://github.com/abgulati/kosmos-2_5-containerized/blob/main/images/docker_wsl_integration.png"  align="center">
        </p>

### 4. Nvidia Container Toolkit

- In a bash shell (search for ```Ubuntu``` in the Start-menu in Windows), perform the following steps:

    - Configure the production repository:

        ```
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        ```

    - Update the packages list from the repository & Install the Nvidia Container Toolkit Packages: 

        ```
        sudo apt-get update && apt-get install -y nvidia-container-toolkit
        ```

    - Configure the container runtime by using the nvidia-ctk command, which modifies the /etc/docker/daemon.json file so that Docker can use the Nvidia Container Runtime:

        ```
        sudo nvidia-ctk runtime configure --runtime=docker
        ```

    - Restart the Docker daemon: 

        ```
        sudo systemctl restart docker
        ```

[Back to Table of Contents](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#table-of-contents)


## Installing & Deploying the Kosmos-2.5 Pre-Built Docker Image

1. Download the pre-built image from my Google Drive (only way I could host it for free!): https://drive.google.com/file/d/18R4XxOmH8R8wwtvhyPM0PCMiO58If2wY/view?usp=sharing

2. Import Image into Docker:

    ```
    docker load -i kosmos_image.tar
    ```

3. Run the Kosmos-2.5 container!

    ```
    docker run --gpus all -p 25000:25000 kosmos-2_5
    ```

[Back to Table of Contents](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#table-of-contents)


## Building the Docker Image

- If for any reason you prefer to build the image yourself rather than using the pre-built image, simply clone the `flash-attention` repo, download the Kosmos-2.5 model checkpoint, and use `docker build`:

    - Navigate to `kosmos-2_5-containerized/kosmos-2_5-container-files`:

        ```
        cd kosmos-2_5-containerized/kosmos-2_5-container-files
        ```

    - Clone the flash-attention repository:

        ```
        git clone https://github.com/Dao-AILab/flash-attention.git
        ```

    - Build:

        ```
        docker build --progress=plain -t kosmos-2_5 .

        # To build without using cached data:
        docker build --progress=plain --no-cache -t kosmos-2_5 .
        ```

[Back to Table of Contents](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#table-of-contents)


## API Specification

1. Endpoint: `/infer`

2. Port: `25000`

3. Header: `Content-Type` header set to `multipart/form-data`

4. Request Body:

    - Type: `form-data`

    - key: `image`

        - Allowed file-types: `png`, `jpg`, `jpeg`, `gif`

    - key: `task`

        - `ocr` for optical-character recognition - outputs text & bounding-box co-ordinates

        - `md` for markdown - outputs text from image in markdown format

5. Response:

    - Format: `json`

    - Successful Response: `{'output': result.stdout, 'error': result.stderr}` 

        - Note: `stderr` contains any warnings - these are not errors but rather general data and `FutureWarning` notifications  

    - Error Response:

        - If `image` key missing:

        Response: `{'error': 'No image file provided'}` 
        Code: `400`

        - If `image` key present but no file sent:

        Response: `{'error': 'No selected file'}` 
        Code: `400`

        - If invalid file-type:

        Response: `{'error': 'Invalid file type'}` 
        Code: `400`

[Back to Table of Contents](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#table-of-contents)


## Invoke Kosmos-2.5 API - /infer endpoint

### via POSTMAN

- Using the Desktop Client:

    - Refer to the screenshot below:
    <p align="center">
    <img src="https://github.com/abgulati/kosmos-2_5-containerized/blob/main/images/POSTMAN-Setup.png"  align="center">
    </p>

    - Open Postman

    - Create a new request: Click on the `+` tab or `New` button to create a new request

    - Set up the request:

        - Change the `HTTP method` to `POST` using the dropdown menu

        - Enter the URL: `http://localhost:25000/infer`

    - Set up the request body:

        - Click on the `Body` tab

        - Select `form-data`

        - You'll need to add two key-value pairs:

            1. For the image:

                ```
                Key: image
                Value: Select "File" from the dropdown next to the key  # click "Select Files" and choose your image file
                ```

            2. For the task:

                ```
                Key: task
                Value: ocr (type this as text)  # or md for markdown
                ```

    - Headers: Postman will automatically set the `Content-Type` header to `multipart/form-data` when you use form-data, so you don't need to set this manually

    - Send the request: Click the "Send" button

### via CURL

- via BASH:

    - For OCR:

        ```
        curl -X POST -F "image=@/path/to/local/image.jpg" -F "task=ocr" http://localhost:25000/infer
        ```

    - For markdown:

        ```
        curl -X POST -F "image=@/path/to/local/image.jpg" -F "task=md" http://localhost:25000/infer
        ```

### via Python Requests

    ```
    import requests

    url = "http://localhost:25000/infer"
    files = {"image": open("path/to/image.jpg", "rb")}
    data = {"task": "ocr"}  # or md for markdown

    response = requests.post(url, files=files, data=data)
    print(response.json())
    ```

### via JavaScript - Fetch

    ```
    const formdata = new FormData();
    formdata.append("image", fileInput.files[0], "path/to/image.jpg");
    formdata.append("task", "ocr"); # or md for markdown

    const requestOptions = {
      method: "POST",
      body: formdata,
      redirect: "follow"
    };

    fetch("http://localhost:25000/infer", requestOptions)
      .then((response) => response.text())
      .then((result) => console.log(result))
      .catch((error) => console.error(error));
    ```

### via JavaScript - jQuery

    ```
    var form = new FormData();
    form.append("image", fileInput.files[0], "path/to/image.jpg");
    form.append("task", "ocr"); # or md for markdown

    var settings = {
      "url": "http://localhost:25000/infer",
      "method": "POST",
      "timeout": 0,
      "processData": false,
      "mimeType": "multipart/form-data",
      "contentType": false,
      "data": form
    };

    $.ajax(settings).done(function (response) {
      console.log(response);
    });
    ```

[Back to Table of Contents](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#table-of-contents)


## Rebuilding the Dependencies & Container - If the Pre-Built Image & dockerfile in this Repo Fail to Work

- If the pre-built image provided in this repository don't work, and neither does a fresh `docker build` with the dockerfile provided, it may be an issue with re-using the [prebuilt wheels](https://github.com/abgulati/kosmos-2_5-containerized/tree/main/kosmos-2_5-container-files/prebuilt_wheels)

- In this case, you may elect to build these dependencies on your system via either of the options below - strap in with your favorite drink (or three) as this is going to be a long ride!


### Option 1 (recommended) - Pre-Build Dependencies in Host Machine & Re-use for `docker build`

- This method is the preferred route as building wheels for the `flash-attention` and `xformers` libraries can take a very long time and exponentially more hardware resources when done via the dockerfile while building the container as compared to doing so on the host system

- For instance, building the `flash-attention` library takes about an hour on my host system (Windows 11, Intel Core i9 13900KF, RTX 3090) while fitting comfortably within the 32GB SysRAM. Within the container build though, it wasn't even half done after an hour and an additional 100GB pagefile was necessary to augment the SysRAM! 

- Even on the host OS, these builds will take a while so don't be alarmed

- If you're doing this on Windows, you must use your Ubuntu-22.04 WSL environment for building the wheels, and then transfer them to Windows to build the container

- In a bash shell (search for ```Ubuntu``` in the Start-menu in Windows), perform the following steps:

1. Install flash-attention:

    - install PyTorch:

        ```
        sudo apt install python3-pip
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124   # modify CUDA version if required - https://pytorch.org/get-started/locally/
        ```

    - install dependencies:

        ```
        pip install wheel==0.37.1
        pip install ninja==1.11.1
        pip install packaging==24.1
        pip install numpy==1.22
        pip install psutil==6.0.0
        ```

    - `git clone` and `cd` repo:

        ```
        git clone https://github.com/Dao-AILab/flash-attention.git
        cd flash-attention
        ```

    - install from repo:

        ```
        pip install . --no-build-isolation
        ```

    - test flash-attention installation (example output: 2.5.9.post1):

        ```
        python3
        import flash_attn
        print(flash_attn.__version__)
        ```

2. Install xformers:

    - Run the below command:

        ```
        pip install git+https://github.com/facebookresearch/xformers.git@04de99bb28aa6de8d48fab3cdbbc9e3874c994b8 
        ```

3. Locate and Copy Wheels:

    - Locate the wheels that were built for the above:

        ```
        find ~ -name "*.whl"
        ```

    - Look for the paths similar to the below:

        ```
        /home/<username>/.cache/pip/wheels/e1/b9/e3/5b5b849d01c0e4007af963f69ad86fb43910a0c18080ee8918/xformers-0.0.22+04de99b.d20240705-cp310-cp310-linux_x86_64.whl

        /home/<username>/.cache/pip/wheels/f6/b4/f5/30df6540ed09f56a99a1138f669e1dbee729478850845504f0/flash_attn-2.5.9.post1-cp310-cp310-linux_x86_64.whl
        ```

    - Copy the wheels:

        - In Linux:

            ```
            cp <path_to_flash_attn_wheel> <path_to_cloned_kosmos_container_repo>/kosmos-2_5-container-files/prebuilt_wheels

            cp <path_to_xformer_wheel> <path_to_cloned_kosmos_container_repo>/kosmos-2_5-container-files/prebuilt_wheels
            ```

        - In Windows:

            - This will make a folder in your Windows User directory:

                ```
                mkdir -p /mnt/c/Users/YourWindowsUsername/wsl_wheels
                ```

            - Transfer wheels from WSL to Windows:

                ```
                cp <path_to_flash_attn_wheel> /mnt/c/Users/YourWindowsUsername/wsl_wheels

                cp <path_to_xformer_wheel> /mnt/c/Users/YourWindowsUsername/wsl_wheels
                ```

            - Now copy the wheels from `C:/Users/YourWindowsUsername/wsl_wheels` to `<path_to_cloned_kosmos_container_repo>/kosmos-2_5-container-files/prebuilt_wheels`

4. Build Container:

    - Download the Kosmos-2.5 model checkpoint to the `<path_to_cloned_kosmos_container_repo>/kosmos-2_5-container-files/kosmos-2_5` directory: https://drive.google.com/file/d/17RwlniqMwbLEMj5ELQd9iQ4kor749Z0e/view?usp=sharing

    - Note: The checkpoint above is the same as the official model checkpoint from: https://huggingface.co/microsoft/kosmos-2.5/resolve/main/ckpt.pt

    - Navigate to `<path_to_cloned_kosmos_container_repo>/kosmos-2_5-container-files`

    - Run Docker Build:

        ```
        docker build --progress=plain -t kosmos-2_5 .

        # To build without using cached data:
        docker build --progress=plain --no-cache -t kosmos-2_5 .
        ```


### Option 2 (very slow) - Build Dependencies Within Container with `docker build`

- WARNING: Very slow and requires significant hardware resources, particularly SysRAM!

- For instance, building the `flash-attention` library takes about an hour on my host system (Windows 11, Intel Core i9 13900KF, RTX 3090) while fitting comfortably within the 32GB SysRAM. Within the container build though, it wasn't even half done after an hour and an additional 100GB pagefile was necessary to augment the SysRAM! 

- If you still chose this route, then you must download the Kosmos-2.5 model checkpoint, clone the `flash-attention` repo, modify the supplied `dockerfile` and use `docker build`:

    - Download the Kosmos-2.5 model checkpoint to the `<path_to_cloned_kosmos_container_repo>/kosmos-2_5-container-files/kosmos-2_5` directory: https://drive.google.com/file/d/17RwlniqMwbLEMj5ELQd9iQ4kor749Z0e/view?usp=sharing

    - Note: The checkpoint above is the same as the official model checkpoint from: https://huggingface.co/microsoft/kosmos-2.5/resolve/main/ckpt.pt

    - Navigate to `kosmos-2_5-containerized/kosmos-2_5-container-files`:

        ```
        cd kosmos-2_5-containerized/kosmos-2_5-container-files
        ```

    - Clone the flash-attention repository:

        ```
        git clone https://github.com/Dao-AILab/flash-attention.git
        ```

- Replace the existing `dockerfile` with the one below (MODIFY IT AS PER COMMENTS!):

    ```
    # Use an official Nvidia CUDA runtime as a parent image - MODIFY CUDA VERSION AS REQUIRED
    FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

    # Avoid interactive prompts - auto-select defaults for any prompts 
    ARG DEBIAN_FRONTEND=noninteractive

    # Set timezone for tzdata package as it's a dependency for some packages
    ENV TZ=America/Los_Angeles

    # Set the working directory in the container
    WORKDIR /app

    # Copy the current directory contents into the container at /app
    COPY . /app

    # Setting environment variables to maximize use of available hardware resources - MODIFY MAX_JOBS AS PER YOUR CPU LOGICAL CORE COUNT
    ENV MAKEFLAGS="-j$(nproc)"
    ENV MAX_JOBS=16

    # OPTIONAL - Change fPIC level, and Set CUDA optimizations as per your GPU arch - `arch=compute_86,code=sm_86` is for RTX 3000 Ampere, `arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90` is for Ampere & Hopper etc
    # ENV CUDA_NVCC_FLAGS="-Xcompiler -fPIC -O3 --use_fast_math -gencode arch=compute_86,code=sm_86"

    # Install Python & PIP
    RUN apt-get update && apt-get install -y python3.10 python3-pip git

    # Install PyTorch Nightly Build for CUDA 12.4, dependencies for Flash Attention 2 and initial dependencies for Kosmos-2.5
    RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 && \
        pip install -v wheel==0.37.1 ninja==1.11.1 packaging==24.1 numpy==1.22 psutil==6.0.0 && \
        pip install -v tiktoken tqdm "omegaconf<=2.1.0" boto3 iopath "fairscale==0.4" "scipy==1.10" triton flask

    # Set work directory to install flash-attention 
    WORKDIR /app/flash-attention

    RUN pip install . --no-build-isolation

    # Change back to the main app directory
    WORKDIR /app

    # Install remaining dependencies for Kosmos-2.5 from custom repos
    RUN pip install -v git+https://github.com/facebookresearch/xformers.git@04de99bb28aa6de8d48fab3cdbbc9e3874c994b8 && \
        pip install -v git+https://github.com/Dod-o/kosmos2.5_tools.git@fairseq && \
        pip install -v git+https://github.com/Dod-o/kosmos2.5_tools.git@infinibatch && \
        pip install -v git+https://github.com/Dod-o/kosmos2.5_tools.git@torchscale && \
        pip install -v git+https://github.com/Dod-o/kosmos2.5_tools.git@transformers

    # Create image upload directory, no error if already exists
    RUN mkdir -p /tmp

    # Make port 25000 available to the world outside this container - MODIFY IF DESIRED
    EXPOSE 25000

    # Change back to the main app directory
    WORKDIR /app/kosmos-2_5

    # Run application
    CMD ["python3", "kosmos_api.py"]
    ```

- Run Docker Build:

    ```
    docker build --progress=plain -t kosmos-2_5 .

    # To build without using cached data:
    docker build --progress=plain --no-cache -t kosmos-2_5 .
    ```

- If you notice `killed` while building flash-attention, your system is resource constrained and the docker process is being killed. To mitigate this, you may try to modify the CPU, RAM, and Pagefile resources allocated to WSL. To do so:

    - Navigate to `C:\Users<YourUsername>`

    - Create a `.wslconfig` file, or modify if it already exists

    - Enter/tweak the below parameters:

        ```
        [wsl2]
        memory=24GB
        processors=16
        swap=80GB
        ```

    - Keep an eye on resource use via the Task Managers `Performance` tab, modify the values above as required

- You may also try to build with increased verbosity as required to diagnose any other issues: `pip install -vvv . --no-build-isolation` 

[Back to Table of Contents](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#table-of-contents)


## Running Kosmos-2.5 Uncontainerized

- If you prefer to setup Kosmos-2.5, be aware that it's incredibly temperamental and has a bunch of specific requirements

- As a result, you may wish you configure a Python virtual environment

- Requirements (in addition to [Dependencies](https://github.com/abgulati/kosmos-2_5-containerized/tree/main?tab=readme-ov-file#dependencies) above)

    - Linux, as `triton` is officially only supported on Linux

        - Use your Ubuntu-22.04 WSL environment if on Windows
    
    - Python v3.10.x, as the custom `fairseq` lib malfunctions on 3.11.x

- In a bash shell (search for ```Ubuntu``` in the Start-menu in Windows), perform the following steps:

    - Install CUDA Toolkit v12.4.1:

        ```
        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin

        sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

        wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb

        sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb

        sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

        sudo apt-get update

        sudo apt-get -y install cuda-toolkit-12-4
        ```

    - Set NVCC PATH:

        - confirm symlink for cuda:

            ```
            ls -l /usr/local/cuda
            ls -l /etc/alternatives/cuda
            ```

        - update bashrc:

            ```
            nano ~/.bashrc

            # add this line to the end of bashrc:
            export PATH=/usr/local/cuda/bin:$PATH
            ```

        - reload bashrc:

            ```
            source ~/.bashrc
            ``` 

    - Confirm CUDA installation:

        ```
        nvcc -V
        nvidia-smi
        ```

    - Install flash-attention:

        - install PyTorch:

            ```
            sudo apt install python3-pip
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
            ```

        - install dependencies:

            ```
            pip install wheel==0.37.1
            pip install ninja==1.11.1
            pip install packaging==24.1
            pip install numpy==1.22
            pip install psutil==6.0.0
            ```

        - `git clone` and `cd` repo:

            ```
            git clone https://github.com/Dao-AILab/flash-attention.git
            cd flash-attention
            ```

        - install from repo:

            ```
            pip install . --no-build-isolation
            ```

        - test flash-attention installation (example output: 2.5.9.post1):

            ```
            python3
            import flash_attn
            print(flash_attn.__version__)
            ```

    - Install Kosmos-2.5!

        - PIP Requirements:

            ```
            pip install tiktoken
            pip install tqdm
            pip install "omegaconf<=2.1.0"
            pip install boto3
            pip install iopath
            pip install "fairscale==0.4"
            pip install "scipy==1.10"
            pip install triton
            pip install git+https://github.com/facebookresearch/xformers.git@04de99bb28aa6de8d48fab3cdbbc9e3874c994b8
            pip install git+https://github.com/Dod-o/kosmos2.5_tools.git@fairseq
            pip install git+https://github.com/Dod-o/kosmos2.5_tools.git@infinibatch
            pip install git+https://github.com/Dod-o/kosmos2.5_tools.git@torchscale
            pip install git+https://github.com/Dod-o/kosmos2.5_tools.git@transformers
            ```

        - Clone Repo and Checkpoint:

            ```
            git clone https://github.com/microsoft/unilm.git

            cd unilm/kosmos-2.5/

            wget https://huggingface.co/microsoft/kosmos-2.5/resolve/main/ckpt.pt
            ```

        - Run OCR!

            ```
            python3 inference.py --do_ocr --image assets/example/in.png -- ckpt ckpt.pt

            python3 inference.py --do_md --image assets/example/in.png -- ckpt ckpt.pt

            ```

[Back to Table of Contents](https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file#table-of-contents)
