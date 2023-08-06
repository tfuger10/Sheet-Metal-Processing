# Part Classification using Neural Networks

***

Tim Fuger

## Project Overview

Our client Fab Inc. is a custom architecture millwork and metalwork manufacturer. They would like to boost the automation and efficiency in the engineering department by automating the process of assigning the first step in the manufacturing process to a part. This project used a dataset of 3D mesh models of 995 different parts supplied by the company to train a Pytorch nueral network to predict manufacturing process assignment for parts. Our final model achieves a high precision and F1-score in the most expensive class (CNC) and can be used immediately by the company to expedite and reduce error in the assignment process.

<img src="Visualizations/Bench2.JPG" width="500" height="500">


## Business Understanding

### Business Problem

> Our stakeholder is looking to **reduce engineering time and error** to **assign the manufacturing process** for a custom fabricated **part**.


Our client, Fab Inc. has already worked to automate large portions of the shop/production floor by having robots and more sophisticated machinery and software work in tandem with shop employees. Since upgrading production equipment and processes, they have seen the bottleneck (backup of work) in their facility start to shift from the production departments to their engineering department. With the increased efficiency in production, the shop floor is always ahead of engineering, which means that the shop is consistently waiting on digital files and shop documentation in order to continue working on projects. Our client would like to boost the automation and reduce error in the engineering department.

To do so, the company has identified that the assignment of manufacturing processes to parts is a task that incurs unnecessary engineering time and overall error. They would like to work with us to analyze and solve this problem.

### Parent Objective

> To develop a **Convolutional Neural Network** that can properly **identify the required processes** for each **3D modeled part.**

### Long Term Goal

This project is the first in a series of planned projects working towards a long term goal of creating a system which can automatically identify and program parts for various assemblies. This in turn will reduce the amount of engineering time required for each project, and will free the team up to focus on other responsibilities. This project is the catalyst and will classify the first step in the manufacturing process for each of the parts.


### Defining Metrics

The metric that is most important in our analysis is precision. We need to be really precise in our predictions, as an imprecise prediction means a material gets sent to the wrong workcell. This, at the least, means lost time for sending the material back to the previous station, but, at the worst, it could mean lost time and lost material if someone cuts stock they should not be cutting. And for the most expensive process, we will want to have that process be as precise. That process will incur the highest time and material cost so we do not want anything being sent to that process that isnt the correct material or part, as it would cost the highest in lost time and material. The company is looking for a better error rate than what it currently has for manufacturing process assignment, and ideally that rate would be above 99%.

Our second metric to use (should there be only slight precision differences between models) will be F1-score. This will help us to balance precision against recall as we analyze the models.


## Data Understanding

### Dataset

The dataset contains 995 models which are all parametric variations on 10 different types of fixtures. These models were collected over the course of a month as they passed from the engineering department to the production department. Each fixture is made up of a variety of parts, with each part having a different shape and a different material assigned. The parts make up assemblies which are identified and shown in the table below.

| Assembly  | Type  |  Picture |
|---|---|---|
| Cab1  | Base Cabinet  | <img src="Visualizations/Cab1.JPG" width="350" height="350"> |
| Cab2  | Wall Cabinet  | <img src="Visualizations/Cab2.JPG" width="350" height="350">|
| Cab3  | Pantry Cabinet  | <img src="Visualizations/Cab3.JPG" width="350" height="350">|
| Shelf1  | Removable Shelf  | <img src="Visualizations/Shelf1.JPG" width="350" height="350">|
| Shelf2  | Floating Shelf  | <img src="Visualizations/Shelf2.JPG" width="350" height="350">|
| Counter1  | Order Counter  | <img src="Visualizations/Counter1.JPG" width="350" height="350">|
| Station1  | Utensil Station  | <img src="Visualizations/Station1.JPG" width="350" height="350">|
| Bench1  | Bench with back  | <img src="Visualizations/Bench1.JPG" width="350" height="350">|
| Bench2  | Bench with no back  | <img src="Visualizations/Bench2.JPG" width="350" height="350">|
| Table1  | Table  | <img src="Visualizations/Table1.JPG" width="350" height="350">|


Each of the solidworks assembly parts were exported to a STL mesh file format using the [Export-to-Mesh](https://github.com/SigmaRelief/SOLIDWORKS-Export-to-Mesh/tree/master) function provided by SigmaRelief.

Samples of these meshes are shown below:

![Sample-of-Meshes](Visualizations/Sample-of-Meshes.png)

Each part has an initial cutting manufacturing process assigned to it. The manufacturing processes are divided into 5 main categories:


| Process  |  Picture |
|---|---|
| CNC Router  | <a href="https://www.homag.com/en/product-detail/cnc-gantry-processing-center-centateq-n-500"><img src="./Visualizations/CNC.jpg"> | 
| Panel Saw  | <a href="https://www.homag.com/en/product-detail/panel-dividing-saw-sawteq-b-300"><img src="./Visualizations/Panel_Saw.jpg">|
| Metal Laser  | <a href="https://www.mazakoptonics.com/machines/optiplex-nexus-3015-fiber/"><img src="./Visualizations/Metal_Laser.png">|
| Metal Band Saw  | <a href="https://www.mscdirect.com/product/details/98049208"><img src="./Visualizations/Metal_Band_Saw.JPG">|
| Waterjet  | <a href="https://www.flowwaterjet.com/waterjet-products/mach-100"><img src="./Visualizations/Waterjet.png">|

    
Image Links:
    
CNC Router - https://www.homag.com/en/product-detail/cnc-gantry-processing-center-centateq-n-500
    
Panel Saw - https://www.homag.com/en/product-detail/panel-dividing-saw-sawteq-b-300
    
Metal Laser - https://www.mazakoptonics.com/machines/optiplex-nexus-3015-fiber/
    
Metal Band Saw - https://www.mscdirect.com/product/details/98049208
    
Waterjet - https://www.flowwaterjet.com/waterjet-products/mach-100

    
### Time and Material Cost Analysis

After a thorough analysis of the time and material estimates supplied by the company, we came to find that the company understandably spends quite a bit of time and material costs in cutting parts through these manufacturing processes we are analyzing.

![Time-Material-Cost](Visualizations/Time-Material-Cost.png)

The engineering time and cost results in 8 hours and 7% assignment error per month. A full day of engineering time is equivalent to 800 dollars and the 7% error rate would be 7,140 dollars in loss from the total in the chart above. This would mean the company has around a 7,940 dollar loss every month from these manufacturing assignment tasks.


## Data Preparation

In order to prepare the data for modeling, we had to complete five distinct tasks that we had identified:

1. Create a dataframe that does not contain duplicate parts to use in modeling.
2. Organize files into folder structure to feed into datasets.
3. Set a standard sample number for amount of vertices in each mesh (the current meshes vary in number of vertices)
4. Create pytorch datasets and dataloaders from the data.
5. Assign class weights to counteract the class imbalance in the dataset.

## Modeling

All of the models produced are based on the PointNet model architecture in order to process the mesh files. The architecture is shown in the original paper, ["PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"](https://arxiv.org/abs/1612.00593) as well as Nikita Karaevv's [Pointnet repo](https://github.com/nikitakaraevv/pointnet) based on the original paper that we used as reference as well.

This approach utilizes a nueral network through pytorch, where all of the data being fed into the model is resampled to have the same size.


Our models include one base model, an augmented data model, and two fine tuned augmented data models. All of the models have a Logsoftmax function in order to generate probabilities for each of the classes of the multiclass classification problem. We will show that the fine tuned model seems to give us the best results due to the additional convolutional layers, dropout layers, and high number of epochs for training.

The base model produced high results on its own, but this was run without any data augmentation applied to the data. This would do poor in situations where data generalization is necessary (for example a shelf at an angle would most likely be classified incorrectly).

![0-Confusion_Matrix](Visualizations/0-Confusion_Matrix.png)

![0-Classification_Report](Visualizations/0-Classification_Report.png)


The final model had normalization and random rotations applied to the Z direction for data transformations for data augmentation. 

![Final-Confusion_Matrix](Visualizations/Final-Confusion_Matrix.png)

![Final-Classification_Report](Visualizations/Final-Classification_Report.png)

The final model performed worse than our base model, but this is to be expected since the base model would not be able to generalize if the same models were rotated in any way other than being oriented along the x, y or z direction (which is currently how all of the data is oriented and does not reflect all real world situations for data that one would receive)


## Evaluation

### Recommendations

The client is looking for an algorithm that can automatically assign parts to different cutting machine processes. While our model does not have the accuracy we would want across all classes, it does get the closest to the precision we are looking for in the most expensive (CNC) class. The ultimate goal of this project is to implement this machine learning algorithm for use in the engineering department. While the algorithm is not ready to be fully implemented, there are some actionable recommendations that the company can take right now:

- Engineers should be required to triple check assigning parts to the CNC class, as this class costs 71% of the total cost for the company.

- The CNC class has the highest time investment of any other class. Though an analysis on production time would be outside the scope of this project, given the amount of time incurred on each machine currently (30 minutes cycle time), it would be beneficial to complete a return on investment analysis of tools and methods that could decrease this time. We are not sure of the tools and methods currently in use, but automatic feed conveyors and material handling systems are just some tools to decrease the cycle time on these machines.

- There are specific features of a part that will dictate what manufacturing process the part might undergo. There seems to be a consistent formula that is followed, so in order to reduce engineering time this formula can be written in a flow diagram which can be used by an administrative member of team who would assign manufacturing processes instead.

### Model Implementation

- The algorithm can be used for a probationary 90 day peiod to automatically assign the CNC designation to parts. It will not be used for any other class or parts during this time. During this time, the model will also be improved by integrating the next steps detailed below. 

- All items designated not CNC will be checked by the engineering team to assign them to the correct processes.

- This model should be improved during the 90 day probationary period, and reevaluated at that time for the precision and F1-score of the model. If the model passes a 99% for the metrics, then the model can be put into full deployment by being applied to classify manufacturing processes for the department.


### Next Steps

Further analysis could yield additional insights such as:

- In the cost material analysis, we would want to calculate the square footage of each individual part and then use that as a basis to determine more accurate costs for each part based on the current dataset

- In the time cost analysis, we would want to implement an analysis program to each machine that would store the part's program and run time for each part's program (depending on the machine, this can include load, unload and machine time), in order to have a more accurate average of time estimates per machine. Those time estimates would then be fed into a program which could estimate program run times for the part models that we have in the dataset.

- Increasing the amount of data that the algorithm trains on would give a larger variety of parts to pull from. While 1,000 3D models is a good start, 10,000 or 20, 000 would be even better for improving the performance of the model.

- Integrating the materials into the modeling would definitely improve the model performance simply because the machines can only cut certain materials.

- The nueral network modeling was accomplished by using the PointNet nueral network architecture. The data would benefit from having another model or two that had a different architecture, just to be able to compare the differences between them. For example, integrating the face normals of the meshes would contribute to improving performance for the model as it would be able to analyze the mesh face information along with the mesh vertex information.

- If possible, since that data is coming from 3d models and not the real world, it would be ideal to have the data in a format that is cleaner if possible such as a step file or other nurbs file. Currently the data science industry does not have a methodology for processing this data due to it being "unstructured".

- After the model has reached a point where full deployment is possible, the next project to undertake would be to handle the sequencing of the assembly of these cut components to start to work on reducing assembly time on the floor.

## Repository Navigation


* [EDA-Modeling-Evaluation](EDA-Modeling-Evaluation.ipynb) contains the main data analysis code and notes for this project.

* [Mesh_Preprocessing_Notebooks/Train-Split-Multiclass-Meshes](Mesh_Preprocessing_Notebooks/Train-Split-Multiclass-Mesh_stl-to-obj.ipynb) contains code which uses a csv file to save meshes (stl) as a different file format (obj) in a new directory according to their classes. The second half of the code will further take the meshes in this newly created directory, and split them into folders for train, validation, and test sets of data.

* Meshes/ folder contains all the meshes used for the dataset. This folder and the contained data can be obtained by downloading the dataset in the Reproducibility instructions below.

 * Meshes/SW_Models/ folder contains all of the original solidworks models that the stl parts were generated from. These models are saved in a Solidworks for makers file format.
 
  * Meshes/SW_Models/Part_Classification-Variations csv file contains all of the different size variations of each of the 10 main assemblies.

  * Meshes/SW_Models/Assembly_Meshes folder contains STL files of each entire assembly of components.

* [Part_Classification](Part_Classification.csv) contains all the 3d model part names, assembly iterations, material, and the manufacturing process identified. This file was created by having a human go through each part and label them accordingly in this csv file.

* [Visualizations/](Visualizations) folder contains all the visualizations from the notebook such as plots and graphs.

* Pytorch_Model_Save/ folder contains all the saved Pytorch model dictionaries, which can be loaded into the model. This folder and the contained data can be obtained by downloading the dataset in the Reproducibility instructions below.

* [Presentation/](Presentation/) folder contains all of the presentation content.

* [Presentation/Part-Processing](Presentation/Part-Processing.pptx) powerpoint presentation.

* [Presentation/Part-Processing](Presentation/Part-Processing.pdf) pdf file of presentation.


## Reproducibility

This repo can be directly downloaded and contains all of the required files for running except for the dataset. The dataset for this project is being hosted on kaggle and needs to be inserted in the following directory in the repository:

```
└── Part-Processing
    └── Meshes
    

└── Part-Processing
    └── Pytorch-Model-Save
            
```

[Part-Processing-Dataset](https://www.kaggle.com/datasets/timfuger/part-processing-dataset)

All specific versions of packages used in this project are located in the conda [environment_pytorch3d.yml](environment_pytorch3d.yml) file located at the top level of this repo.


For additional info, contact Tim Fuger at tfuger10@gmail.com