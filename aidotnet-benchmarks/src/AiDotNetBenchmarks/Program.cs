using Microsoft.AspNetCore.Http.Features;

// Benchmarks run through controllers, matching this repo's format:
//   POST /api/Benchmark/Models     — four-model AiDotNet benchmark (JSON report)
//   POST /api/Regression/...       — regression lifecycle benchmarks
//   POST /api/Both/...             — fan-out to this host AND the PyTorch host
// The benchmark implementation lives under Benchmarks/ (BenchmarkRunner,
// BenchmarkModels); this file is only the web-host bootstrap.

var builder = WebApplication.CreateBuilder(args);

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = long.MaxValue; // unlimited
});

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = long.MaxValue;
});


builder.Services.AddControllers();

var app = builder.Build();

app.MapControllers();

app.Run();
