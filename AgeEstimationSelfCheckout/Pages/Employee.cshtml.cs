using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Text;
using System.Text.Json;
using static AgeEstimationSelfCheckout.Pages.ScanModel;

namespace AgeEstimationSelfCheckout.Pages
{
    public class EmployeeModel : PageModel
    {
        [BindProperty]
        public bool? EmployeeChoice { get; set; } = null;
        public static Data Data { get; set; } = new();

        public async Task OnGetAsync()
        {
            var resultJson = TempData["PredictionResult"] as string;
            if (!string.IsNullOrEmpty(resultJson))
            {
                Data = JsonSerializer.Deserialize<Data>(resultJson);
                if (!Data.AutomaticAgeVerification)
                {
                    using var client = new HttpClient();
                    var content = new StringContent("{}", Encoding.UTF8, "application/json");
                    var response = await client.PostAsync("http://localhost:5000/api/delete", content);
                }
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
