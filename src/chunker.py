def chunk_texts(texts, metas, chunk_size=1000, chunk_overlap=200):
    all_chunks, all_metas = [], []
    for text, meta in zip(texts, metas):
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            all_chunks.append(chunk)
            all_metas.append(meta)
            start = end - chunk_overlap
            if start < 0:
                start = 0
            if end == len(text):
                break
    return all_chunks, all_metas
