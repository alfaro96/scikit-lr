.. toctree::
   :hidden:

   Install <install>
   getting_started
   user_guide
   API <api/index>
   auto_examples/index
   whats_new
   Development <developers/index>

scikit-lr
=========

scikit-lr is a Python package for label ranking and partial label ranking,
built on top of `scikit-learn <https://scikit-learn.org>`_ and distributed
under the MIT license.

In label ranking, each sample is associated with a ranking of a fixed set of
labels instead of a single class, and the goal is to learn a model that
predicts that ranking for new samples. Partial label ranking extends the
problem to rankings with ties. scikit-lr provides estimators and metrics for
both problems that follow the scikit-learn API, so they can be used with
tools such as pipelines, cross-validation and hyperparameter search.
