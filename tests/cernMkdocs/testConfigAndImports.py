from pathlib import Path
from tools.cernMkdocs.service import CernDocsService

config = Path.home() / ".config/heptapod/cern-docs.json"
print("Config exists:", config.is_file())

service = CernDocsService(config)

print("Sources:", list(service.sources))
print("Indexes:", list(service.indices))

for source_id, source in service.sources.items():
    print(
        f"{source_id}: type={source.source_type}, "
        f"site={source.docs_site_url}, "
        f"index={source.search_index_url}"
    )
