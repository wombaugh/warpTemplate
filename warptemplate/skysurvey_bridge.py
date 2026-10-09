"""
Bridge between warpTemplate (github.com/wombaugh/warpTemplate) warped
sncosmo templates and skysurvey (github.com/MickaelRigault/skysurvey)
Target/Transient objects.

warpTemplate.loaders.WarpfitTemplateLoader.get_templates() returns a *flat
list* of already-instantiated sncosmo.Model objects (one per drawn
SN-basis/warp-fit combination, possibly with repeats since sampling is done
with replacement), each carrying a relative sampling weight
("template_prob"). skysurvey, on the other hand, wants a Target that knows
how to *generate itself*: either a single template (TSTransient) or a small
set of named templates it draws among at simulation time
(MultiTemplateTSTransient -- exactly what it already uses internally for
e.g. the 23 Vincenzi+19 SNe II templates).

WarpTemplatePopulation is the glue: it turns a warpTemplate template list
into a de-duplicated, uniquely-named skysurvey.TemplateCollection plus a
matching per-template rate array, wrapped in a MultiTemplateTSTransient
subclass so the result can be handed directly to
DataSet.from_targets_and_survey, list-combined via skysurvey.TargetCollection,
etc.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Optional, Sequence, Union

import numpy as np
import sncosmo

from skysurvey.target.timeserie import MultiTemplateTSTransient
from skysurvey.template import TemplateCollection

__all__ = ["WarpTemplatePopulation"]


# --------------------------------------------------------------------- #
#  helpers                                                               #
# --------------------------------------------------------------------- #
def _as_model_weight_pairs(warp_templates, weight_key="template_prob"):
    """Normalize warpTemplate.get_templates() output -- or a bare list of
    sncosmo.Model / sncosmo.Source objects -- into (sncosmo.Model, weight)
    pairs.
    """
    pairs = []
    for item in np.atleast_1d(warp_templates):
        if isinstance(item, dict):
            model = item.get("model")
            if model is None:
                raise ValueError(
                    "warp_templates dict entry has no 'model' key "
                    f"(got keys={list(item.keys())}); is this really "
                    "WarpfitTemplateLoader.get_templates() output?"
                )
            weight = float(item.get(weight_key, 1.0) or 1.0)
        else:
            model = item
            weight = 1.0

        if isinstance(model, sncosmo.Source):
            model = sncosmo.Model(model)
        elif not isinstance(model, sncosmo.Model):
            raise TypeError(
                f"Unsupported template type {type(model)!r}; expected an "
                "sncosmo.Model, sncosmo.Source, or a "
                "WarpfitTemplateLoader.get_templates() dict."
            )
        pairs.append((model, weight))
    return pairs


def _dedupe_by_identity_and_name(pairs):
    """Collapse (model, weight) pairs that are literally the same object --
    this happens whenever warpTemplate's with-replacement sampling drew the
    same warp fit more than once -- by summing their weights. Then make
    sure the surviving sncosmo source names are unique, since skysurvey
    identifies/looks up templates by `source.name`.
    """
    if not pairs:
        # A legitimate, documented outcome of WarpfitTemplateLoader.get_templates()
        # (e.g. nothing survives a strict min_fit_quality filter) -- not an error.
        return [], np.asarray([], dtype=float)

    by_id = OrderedDict()
    for model, weight in pairs:
        key = id(model)
        if key in by_id:
            existing_model, existing_weight = by_id[key]
            by_id[key] = (existing_model, existing_weight + weight)
        else:
            by_id[key] = (model, weight)

    models, weights = zip(*by_id.values())
    models = list(models)
    weights = np.asarray(weights, dtype=float)

    names = [m.source.name for m in models]
    seen = {}
    unique_names = []
    for name in names:
        n = seen.get(name, 0)
        seen[name] = n + 1
        unique_names.append(name if n == 0 else f"{name}__dup{n}")

    if unique_names != names:
        dupes = sorted({n for n in names if names.count(n) > 1})
        warnings.warn(
            f"{len(dupes)} template name(s) were shared by *different* "
            f"sncosmo models ({dupes[:5]}{'...' if len(dupes) > 5 else ''}); "
            "renamed them to keep skysurvey's per-template bookkeeping "
            "unique. Give warpTemplate models unique names to avoid this."
        )
        for model, new_name in zip(models, unique_names):
            if model.source.name != new_name:
                model.source.name = new_name

    return models, weights


# --------------------------------------------------------------------- #
#  main class                                                            #
# --------------------------------------------------------------------- #
class WarpTemplatePopulation(MultiTemplateTSTransient):
    """A skysurvey MultiTemplateTSTransient populated from warpTemplate output.

    Examples
    --------
    >>> from warptemplate.loaders import WarpfitTemplateLoader
    >>> loader = WarpfitTemplateLoader("/path/to/warpcoeffs")
    >>> sniip = WarpTemplatePopulation.from_warp_loader(
    ...     loader, fitclass="SNII", rate=7.0e4, magabs=(-16.75, 1.0),
    ...     get_templates_kwargs=dict(template_selection="all",
    ...                               snbasis_selection="all",
    ...                               min_fit_quality="silver"),
    ... )
    >>> sniip.draw(size=20_000, tstart=58_000, tstop=58_365, inplace=True)
    >>> from skysurvey import DataSet
    >>> dset = DataSet.from_targets_and_survey(sniip, survey)

    Or, if you already called `get_templates()` yourself:

    >>> warp_templates = loader.get_templates("SNII", template_selection=2)
    >>> pop = WarpTemplatePopulation(warp_templates, magabs=(-16.75, 1.0))
    >>> pop.set_rate(pop.weighted_rate(7.0e4))
    >>> pop.draw(size=20_000, tstart=58_000, tstop=58_365, inplace=True)
    """

    _RATE = 1.0  # arbitrary placeholder; always overridden via rate=/weighted_rate()

    def __init__(self, warp_templates=None, weight_key="template_prob",
                 magabs=None, **kwargs):
        """
        Parameters
        ----------
        warp_templates : list, optional
            Output of `WarpfitTemplateLoader.get_templates(...)` (a list of
            dicts, each with at least a "model" key holding an
            `sncosmo.Model`), or a bare list of `sncosmo.Model`/
            `sncosmo.Source` objects. May be None (set later via
            `set_template`).
        weight_key : str
            Key of each `warp_templates` dict used as *relative* sampling
            weight between templates (default: "template_prob", as returned
            by `get_templates()`). Ignored for bare model/source lists
            (all such templates are equiprobable).
        magabs : tuple, optional
            (loc, scale) or (loc, scale_low, scale_high); same convention as
            `skysurvey.TSTransient`.
        **kwargs
            Forwarded up to `Target.__init__` (generally unused directly,
            but keeps the class compatible with `Target.from_draw`).
        """
        self._template_weights = None
        if warp_templates is not None:
            self.set_template(warp_templates, weight_key=weight_key)
        super().__init__(template=None, magabs=magabs, **kwargs)

    @classmethod
    def _parse_init_kwargs_(cls, **kwargs):
        """ trick to add specific subclass kwargs into the init (see
        skysurvey.target.core.Target._parse_init_kwargs_) """
        init_kwargs = {
            "warp_templates": kwargs.pop("warp_templates", None),
            "weight_key": kwargs.pop("weight_key", "template_prob"),
        }
        magabs_kwargs, kwargs = super()._parse_init_kwargs_(**kwargs)
        init_kwargs |= magabs_kwargs
        return init_kwargs, kwargs

    # ------------------------------------------------------------------ #
    #  template / rate wiring                                            #
    # ------------------------------------------------------------------ #
    def set_template(self, warp_templates, weight_key="template_prob",
                      force_uniquetype=True):
        """Load a warpTemplate template list (or bare sncosmo models/sources).

        Overrides `MultiTemplateTSTransient.set_template`: instead of
        expecting an already-unique list of named sncosmo sources, this
        accepts the raw (possibly repeat-sampled, dict-wrapped) output of
        `WarpfitTemplateLoader.get_templates()`, de-duplicates it, and
        derives the corresponding per-template relative weights.
        """
        pairs = _as_model_weight_pairs(warp_templates, weight_key=weight_key)
        models, weights = _dedupe_by_identity_and_name(pairs)

        templatecol = TemplateCollection.from_sncosmo(models)
        if force_uniquetype and templatecol.ntemplates > 0 and not templatecol.is_uniquetype:
            raise ValueError(
                "warp_templates mix multiple sncosmo.Source subclasses; "
                "this is not supported by skysurvey's MultiTemplateTSTransient "
                "(force_uniquetype=True)."
            )

        self._template = templatecol
        self._template_weights = (
            weights / weights.sum() if weights.size and weights.sum() > 0
            else weights.astype(float)
        )

    def weighted_rate(self, total_rate):
        """Return `total_rate` unchanged (kept as a scalar volumetric rate).

        Note
        ----
        skysurvey (as of 1.0.0) also lets `MultiTemplateTSTransient.set_rate`
        take a *list* of per-template rates for weighted template drawing --
        but that code path is currently broken (it does not correctly
        broadcast the per-template rate array against the drawn sample size
        in `draw_template`, see `skysurvey/target/timeserie.py`). To stay
        correct regardless of that, this class keeps `self.rate` a plain
        scalar and instead applies the warpTemplate-derived per-template
        weights itself in `draw_template` (see below). This method is kept
        (returning `total_rate` as-is) so the call site doesn't need to
        change if/when upstream fixes the multi-rate broadcasting.
        """
        if self._template_weights is None:
            raise RuntimeError("call set_template() first.")
        return float(total_rate)

    def draw_template(self, size=None, redshift=None, rng=None):
        """Draw a template name per target, respecting each warped
        template's relative sampling weight (`self._template_weights`).

        Overrides `MultiTemplateTSTransient.draw_template` to avoid relying
        on skysurvey's (currently broken, see `weighted_rate`) per-template
        rate-array broadcasting; weights are applied directly here instead.
        """
        if self._template_weights is None:
            return super().draw_template(size=size, redshift=redshift, rng=rng)

        size = len(redshift)
        rng = np.random.default_rng(rng)
        return rng.choice(self.template.names, size=size, p=self._template_weights)

    # ------------------------------------------------------------------ #
    #  convenience constructor                                           #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_warp_loader(cls, loader, fitclass, rate=1.0, magabs=None,
                          get_templates_kwargs=None,
                          size=None, draw_kwargs=None, **kwargs):
        """Build (and optionally draw) a WarpTemplatePopulation directly
        from a warpTemplate loader.

        Parameters
        ----------
        loader : warptemplate.loaders.WarpfitTemplateLoader
        fitclass : str
            Forwarded to `loader.get_templates(fitclass, **get_templates_kwargs)`.
        rate : float
            Total population volumetric rate (events / Gpc3 / yr);
            distributed across the drawn templates proportionally to their
            sampling weight (see `weight_key`/`weighted_rate`).
        magabs : tuple, optional
            See `WarpTemplatePopulation.__init__`.
        get_templates_kwargs : dict, optional
            Forwarded to `loader.get_templates()`, e.g. `template_selection`,
            `snbasis_selection`, `min_fit_quality`, `color_mode`,
            `random_seed`, ...
        size : int, optional
            If given, immediately calls `.draw(size=size, inplace=True,
            **draw_kwargs)` before returning.
        draw_kwargs : dict, optional
            Extra kwargs for `.draw()` (`tstart`, `tstop`, `zmax`, `skyarea`,
            ...), only used if `size` is given.
        **kwargs
            Forwarded to the constructor.

        Returns
        -------
        WarpTemplatePopulation
        """
        warp_templates = loader.get_templates(fitclass, **(get_templates_kwargs or {}))
        if not warp_templates:
            raise ValueError(
                f"loader.get_templates({fitclass!r}, ...) returned no templates "
                "(check your filtering options)."
            )

        this = cls(warp_templates=warp_templates, magabs=magabs, **kwargs)
        this.set_rate(this.weighted_rate(rate))

        if size is not None:
            this.draw(size=size, inplace=True, **(draw_kwargs or {}))

        return this
