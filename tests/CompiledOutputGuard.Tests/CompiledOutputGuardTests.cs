using Xunit;

public class CompiledOutputGuardTests
{
    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void RejectsNonFiniteValuesOnEitherSide(float value)
    {
        Assert.False(CompiledOutputGuard.Matches([1], [value], [1], [1]));
        Assert.False(CompiledOutputGuard.Matches([value], [1], [1], [1]));
        Assert.False(CompiledOutputGuard.Matches([value], [value], [1], [1]));
    }

    [Fact]
    public void RejectsShorterAndLongerOutputsBeforeIndexing()
    {
        Assert.False(CompiledOutputGuard.Matches([1, 2], [1], [2], [1]));
        Assert.False(CompiledOutputGuard.Matches([1], [1, 2], [1], [2]));
        Assert.False(CompiledOutputGuard.Matches([], [], [0], [0]));
    }

    [Fact]
    public void RejectsSameValuesWithDifferentGeometry() =>
        Assert.False(CompiledOutputGuard.Matches([1, 2, 3, 4], [1, 2, 3, 4], [2, 2], [1, 4]));

    [Fact]
    public void RejectsFiniteCorruption() =>
        Assert.False(CompiledOutputGuard.Matches([1, -2], [1, 3], [1, 2], [1, 2]));

    [Fact]
    public void AcceptsMatchingAndNumericallyCloseFiniteOutputs()
    {
        Assert.True(CompiledOutputGuard.Matches([0, 0], [0, 0], [2], [2]));
        Assert.True(CompiledOutputGuard.Matches([1, -2], [1.00001f, -2], [1, 2], [1, 2]));
    }
}
