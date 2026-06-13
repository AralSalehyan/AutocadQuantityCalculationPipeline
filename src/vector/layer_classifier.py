from unicodedata import normalize


class LayerClassifier:
    def classify(self, layer_name: str | None) -> str:
        if not layer_name:
            return "unknown"
        normalized = _normalize(layer_name)
        if any(token in normalized for token in ("wall", "walls", "duvar", "a-wall", "a_walls", "architectural")):
            return "wall"
        if any(token in normalized for token in ("kapi", "door")):
            return "door"
        if any(token in normalized for token in ("pencere", "window", "win")):
            return "window"
        if any(token in normalized for token in ("text", "txt", "room", "oda", "salon")):
            return "text"
        if any(token in normalized for token in ("dim", "dimension", "olcu")):
            return "dimension"
        return "unknown"


def _normalize(value: str) -> str:
    value = value.replace("\u0131", "i").replace("\u0130", "I")
    decomposed = normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return decomposed.lower()
