:mod:`{{ module }}`.{{ objname }}
{{ underline }}========

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}
    .. rubric:: Methods

    {% if methods %}
    .. autosummary::
    {% for item in methods %}
        ..
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}

    .. automethod:: __init__
    {% endblock %}
