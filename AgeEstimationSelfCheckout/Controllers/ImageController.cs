using Microsoft.AspNetCore.Mvc;
using System.IO;
using System.Threading.Tasks;

namespace AgeEstimationSelfCheckout.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ImageController : ControllerBase
    {
        [HttpPost("upload")]
        public async Task<IActionResult> UploadImage([FromBody] ImageUploadRequest request)
        {
            if (request == null || string.IsNullOrWhiteSpace(request.ImageData) || string.IsNullOrWhiteSpace(request.ImageName))
            {
                return BadRequest("Invalid image data.");
            }

            var base64Data = request.ImageData.Substring(request.ImageData.IndexOf(",") + 1);
            var imageBytes = Convert.FromBase64String(base64Data);

            var directoryPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot/images");
            if (!Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }

            var filePath = Path.Combine(directoryPath, request.ImageName);
            await System.IO.File.WriteAllBytesAsync(filePath, imageBytes);

            return Ok(new { message = "Image uploaded successfully." });
        }

        public class ImageUploadRequest
        {
            public string ImageData { get; set; }
            public string ImageName { get; set; }
        }
    }
}
