using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Text.Json;
using static AgeEstimationSelfCheckout.Pages.ScanModel;

namespace AgeEstimationSelfCheckout.Pages
{
    public class EmployeeModel : PageModel
    {
        [BindProperty]
        public bool? EmployeeChoice { get; set; } = null;

        [BindProperty]
        public static Data Data { get; set; } = new();

        public void OnGet()
        {
            var resultJson = TempData["PredictionResult"] as string;
            if (!string.IsNullOrEmpty(resultJson))
            {
                Data = JsonSerializer.Deserialize<Data>(resultJson);
            }
        }
        public async Task<IActionResult> OnPostAsync()
        {
            if (Request.Form["EmployeeChoice"] == "true")
            {
                EmployeeChoice = true;
            }
            else if (Request.Form["EmployeeChoice"] == "false")
            {
                EmployeeChoice = false;
            }
            return Page();
        }
    }
}
