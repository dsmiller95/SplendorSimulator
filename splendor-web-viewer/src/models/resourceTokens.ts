export function GetTokenFileName(tokenIndex: number) : string {
    const resources = [
        "Ruby",
        "Emerald",
        "Sapphire",
        "Diamond",
        "Onyx",
        "Gold",
    ]
    return resources[tokenIndex]
}