"""interactive chemical viewer

Install
-------
.. code-block:: bash
    pip install chemical-viewer


How to use
----------

CLI
^^^

.. code-block:: bash

    chemical-viewer # see 'chemical-viewr --help'


Python-API
^^^^^^^^^^

.. code-block:: python

    from chemical_viewer import InteractiveChemicalViewer
    # when jupyter, `%matplotlib widgets` has needed.

    viewer = InteractiveChemicalViewer()
    viewer.scatter([1], [1], mols=[Chem.MolFromSmiles("CC")], texts=["test"])
    plt.show()


"""

__version__ = "0.0.5alpha0"
__license__ = "MIT"
__author__ = "yu9824"
__copyright__ = "Copyright © 2024 yu9824"


from chemical_viewer.viewer._core import InteractiveChemicalViewer

__all__ = ("InteractiveChemicalViewer",)
