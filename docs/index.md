# Welcome to intellikit


[![image](https://img.shields.io/pypi/v/intellikit.svg)](https://pypi.python.org/pypi/intellikit)


**A python toolkit for case based reasoning, information retrieval, natural language processing and other techniques for AI and intelligent systems.**

**“intellikit” i.e. Intelligent Tool Kit** is a toolkit for **Case Based Reasoning** (CBR) and **Information Retrieval** (IR) in python. This package is being built primarily for educational purposes, and some content in it may be done more efficiently using Scikit-Learn and other libraries. In some instances such library functions are added directly in “intellikit” but feel free to test out those libraries concurrently and choose what suits your needs best. Some rare similarity measures are implemented from scratch in intellikit but you can extend the functions or implement your own functions depending on your needs. 

In case you need help getting started, the website for this library can be [accessed here!](https://ArthurKakande.github.io/intellikit) Multiple demo projects are added to the examples tab.

*If you are new to Case Based Reasoning and Information Retrieval entirely, here a simple refresher for you:*

**Case-Based Reasoning (CBR)** is a methodology for solving problems. These 
problems may be of a variety of natures. In principle, no problem type is excluded from 
being solved with the CBR methodology. The problem types range from exact sciences 
to mundane tasks. However, this does not mean that CBR is recommended for all problems.

Experiences are essential for CBR. In general, an experience is a recorded episode that occurred in the past, such as “Remember, last time a patient came in with similar symptoms, they had a particular infection” and such experiences are used to help solve future problems or make future decisions. Cases are experiences, they have a context and they also include problems and solutions. A case is explicitly represented/organized using case representations. These can be for example; 

-   **Feature-value pairs.** A feature value pair is used to represent a state of an entity, for example, colour of an entity, “Jessica’s car is red”, where the feature is the colour of the car and the value is red, and the entity is Jessica’s car.
-   **Textual case representation** (for this representation we consider elements of information retrieval)
-   **Object-oriented case representations**
-   **Graph-based case representations**

A key important aspect of CBR is similarity and retrieval. The purpose of retrieval is to retrieve the case from the case base (i.e., a candidate case) that is so similar to a given new problem that their solutions can be swapped. 


-   Free software: MIT License
-   Documentation: <https://ArthurKakande.github.io/intellikit>
    

## Feature_Value Pairs
## Textual case representation
## Object-oriented case representation
## Graph-based case representation

## Text/String Attributes Similarity
-   Hamming distance
-   Hamming Similarity
-   Levenshtien distance
-   Levenshtien similarity
-   Level similarity
-   N-grams

## Document Retrieval (Information Retrieval)
-   Cosine similarity using Vector space model (TF - IDF) 
-   Cosine similarity using Okapi BM25
-   Cosine similarity using Sentence Transformers

## Numeric Attribute Similarity
-   City block metric
-   Euclidean distance
-   Weighted euclidean distance

## Upcoming Features that will be added in upcoming releases
-   A question answering module.
-   CBR similarity measures for taxonomies.
-   CBR measures and examples for Object oriented case representations and Graph based representations and Ontologies.

