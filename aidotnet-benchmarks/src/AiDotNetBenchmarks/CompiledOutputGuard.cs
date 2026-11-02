internal static class CompiledOutputGuard
{
    internal static bool Matches(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual,
        ReadOnlySpan<int> expectedShape, ReadOnlySpan<int> actualShape)
    {
        if (expected.IsEmpty || expected.Length != actual.Length || !expectedShape.SequenceEqual(actualShape))
            return false;
        double maxError = 0, maxMagnitude = 1e-6;
        for (int i = 0; i < expected.Length; i++)
        {
            if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i])) return false;
            maxError = Math.Max(maxError, Math.Abs((double)expected[i] - actual[i]));
            maxMagnitude = Math.Max(maxMagnitude, Math.Abs(expected[i]));
        }
        return maxError / maxMagnitude <= 1e-3;
    }
}
