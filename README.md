# PicExifXML

## Overview

PicExifXML is a tool designed for automatically generating metadata and descriptions for photos, which are then saved in an XML format. The project leverages EXIF data from photographs to create rich, descriptive metadata and integrates AI technology to generate textual descriptions of the photos. This tool is perfect for photographers, archivists, and developers looking to enrich their photo collections with detailed information and automatic descriptions.

## Features

- **EXIF Data Reading**: PicExifXML analyzes `.jpg` images to read and extract EXIF data.
- **Description Generation**: Utilizing AI technology, the tool generates descriptions for each photo, providing context and descriptive content.
- **XML File Creation**: Based on the collected data and generated descriptions, PicExifXML creates XML files containing all this information.
- **Optional Description Improvements**: Users have the option to add an improved description for a photo if the automatically generated one is insufficient.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Dar-Na/PicExifXML.git

2. **Install dependencies**
    
   ```bash
    pip install -r requirements.txt

3. **Running**
    
    Run main function