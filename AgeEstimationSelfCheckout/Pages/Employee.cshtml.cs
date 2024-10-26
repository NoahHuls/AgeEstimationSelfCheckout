using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace AgeEstimationSelfCheckout.Pages
{
    public class EmployeeModel : PageModel
    {
        [BindProperty]
        public bool? EmployeeChoice { get; set; } = null;
        public int ModelAgeEstimation { get; set; } = -1;

        public void OnGet()
        {
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
