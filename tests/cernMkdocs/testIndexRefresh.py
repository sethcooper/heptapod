from pathlib import Path
from tools.cernMkdocs.service import CernDocsService

config = Path.home() / ".config/heptapod/cern-docs.json"
service = CernDocsService(config)

print("Sources:", list(service.sources))
print("Indexes:", list(service.indices))

print("Refreshing indexes...")
service.initialize()

for source_id, index in service.indices.items():
    print(
        f"{source_id}: loaded={index.is_loaded}, "
        f"documents={len(index.docs)}, stale={index.is_stale}"
    )
