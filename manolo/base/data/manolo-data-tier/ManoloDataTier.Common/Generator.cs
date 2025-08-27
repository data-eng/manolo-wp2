namespace ManoloDataTier.Common;

public static class Generator{

    public static string GenerateUlidWithSuffix(string abbreviation) =>
        $"{Ulid.NewUlid()}{abbreviation}";

}