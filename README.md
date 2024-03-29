<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/andy-bridger/stemsegment">
    <img src="images/stemseg.png" alt="Logo" width="140" height="140">
  </a>

<h3 align="center">STEMseg</h3>

  <p align="center">
    Help the processing of 4D-STEM data, specifically automatically clustering similar pixels in SEND pencil beam data 
    <br />
    <a href="https://github.com/andy-bridger/stemsegment/tree/main/example-notebooks">View Demo</a>
    ·
    <a href="https://github.com/andy-bridger/stemsegment/issues">Report Bug</a>
    ·
    <a href="https://github.com/andy-bridger/stemsegment/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#running-on-the-dls-jupyterhub">Running on DLS-Jupyterhub</a>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Intended for processing 4D-STEM datasets using an autoencoder trainer and a DataWrapper for processing the samples. The example notebooks should walkthrough the process of training the encoder and using it to cluster regions with similar diffraction data automatically.



<!-- RUNNING ON DLS-JUPYTERHUB -->
## Running on the DLS Jupyterhub

- Make a copy of the example-notebooks

- Copy the folders stemseg and stemutils (under modules folder) to the notebook's local directory or /home/{YOUR FED ID}/.local/lib/python3.7/site-packages 

- Boot up a GPU kernel having replaced the container image with: CONTAINER_IMAGE=gcr.io/diamond-pubreg/container-tools/jhub-notebook:cuda10.1

- Select the EPSIC3.7 environment to run the example notebooks (if not a default)

- First work through TrainModels.ipynb to create a trained VAE for your datasets

- Then work through TrainInvestigator.ipynb to use the model to cluster


<!-- CONTACT -->
## Contact

Andy Bridger - andrew.bridger@univ.ox.ac.uk

Project Link: [https://github.com/andy-bridger/stemsegment](https://github.com/andy-bridger/stemsegment)

<p align="right">(<a href="#top">back to top</a>)</p>
