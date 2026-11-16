"""Page restoration for the books digitalizer: raw scan frames -> cleaned leaves (evilflowers-scan-service#1, #2).

- ``tasks``: ``evilflowers_scan.clean`` and ``evilflowers_scan.erase``, standalone jobs (evilflowers-worker#5);
- ``jobs``: what each task does, from the validated message to the outcome file (written last);
- ``contract``: the messages and the outcome, as the digitalizer's orchestrator sends and reads them;
- ``imaging``: the ScanTailor cleaner, DocRes, erasures, WebP previews and Deep Zoom pyramids;
- ``artefacts``: marks on a leaf (stamps, stickers, bleed-through, scan edges) with a band each.
"""
