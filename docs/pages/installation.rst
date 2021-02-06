Installation
============

Linux

1. Install python 3.7+

Windows Steps

1. Download `anaconda`_ 3
2. Start > Anaconda Prompt (anaconda)

Common Steps

0. Install Dota2

1. ``git clone https://github.com/Delaunay/LuaFun.git``
2. ``cd LuaFun``

3. Install Dependencies

.. code-block:: bash

    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    pip install -r requirements.txt
    pip install -e .

4. Change the bot folder

.. code-block:: bash

    DOTA_PATH=/media/setepenre/local/SteamLibraryLinux/steamapps/common/dota\ 2\ beta
    ln -f -s botslua/ $DOTA_PATH/game/dota/scripts/vscripts/bots

5. Run the simulation

6. Render the game or use you `browser`_ to see how it is going

.. image:: ../_static/StateOverview.png


.. _browser: http://localhost:5000/draw/radiant
.. _anaconda: https://www.anaconda.com/products/individual
