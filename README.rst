======
T-Fold
======

An open-source, low code python package for the implementation of the *T-Fold Sequential Validation (T-Fold SV)* method, which is aimed to be the go to method in subsitution to any variation of *K-Fold Cross Validation (K-Fold CV)* method, for the case of Financial Time Series data, specially in a predictive modeling process.

-------------
Documentation
-------------

- Github repository: https://github.com/iffranciscome/T-Fold-SV

------------
Installation
------------

- Cloning repository
  
Clone entire github project

    git@github.com:IFFranciscoME/T-Fold-SV.git

(optional) create a virtual environment

    virtualenv venv

(optional) activate virtual environment

        source ~/venv/bin/activate

and then install dependencies

        pip install -r requirements.txt

------
Author
------

J.Francisco Munnoz - `IFFranciscoME`_ - Is an Associate Professor in the Mathematics and Physics Department, at `ITESO`_ University.

.. _ITESO: https://iteso.mx/
.. _IFFranciscoME: https://iffranciscome.com/


--------------------
Current Contributors
--------------------

.. image:: https://contrib.rocks/image?repo=IFFranciscoME/T-Fold-SV
        :target: https://github.com/IFFranciscoME/T-Fold-SV/graphs/contributors
        :alt: Contributors

-------
License
-------

**GNU General Public License v3.0** 

*Permissions of this strong copyleft license are conditioned on making available 
complete source code of licensed works and modifications, which include larger 
works using a licensed work, under the same license. Copyright and license notices 
must be preserved. Contributors provide an express grant of patent rights.*

*Contact: For more information in reggards of this project, please contact francisco.me@iteso.mx*

----------
LaTeX Test
----------

$\hat{y_{t}} = \gamma_{t} + \sum_{t=0}^{T} \frac{1}{\beta7}$

\begin{equation}
    \hat{y_{t}} = \gamma_{t} + \sum_{t=0}^{T} \frac{1}{\beta7}
\end{equation}
