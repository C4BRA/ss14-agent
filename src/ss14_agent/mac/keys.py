"""macOS virtual key codes (US ANSI layout) for the keys the agent uses."""

KEYCODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29,
    "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    ";": 41, ",": 43, ".": 47, "/": 44,
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "escape": 53,
    "shift": 56, "ctrl": 59, "control": 59, "alt": 58, "option": 58, "cmd": 55,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "left": 123, "right": 124, "down": 125, "up": 126,
}

MODIFIER_KEYS = {"shift", "ctrl", "control", "alt", "option", "cmd"}


def keycode(key: str) -> int:
    try:
        return KEYCODES[key.lower()]
    except KeyError:
        raise ValueError(f"No key code for '{key}'") from None
