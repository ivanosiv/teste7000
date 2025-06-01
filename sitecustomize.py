# sitecustomize.py — esse arquivo é executado automaticamente antes de qualquer outro import
import collections
import collections.abc

# Garante que qualquer biblioteca que tente usar collections.MutableMapping
# vá encontrar a classe em collections.abc.MutableMapping
collections.MutableMapping = collections.abc.MutableMapping
