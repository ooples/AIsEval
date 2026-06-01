using AiDotNet;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Http.Features;
using System.Diagnostics;
using System.Text.Json;

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
