# general imports
from pathlib import Path
import numpy as np

# AHA imports
import magma as m
import fault

# svreal import
from svreal import get_svreal_header

# msdsl imports
from ..common import pytest_sim_params, get_file
from msdsl import MixedSignalModel, VerilogGenerator, get_msdsl_header

BUILD_DIR = Path(__file__).resolve().parent / 'build'
DOMAIN = np.pi
RANGE = 1.0

def pytest_generate_tests(metafunc):
    pytest_sim_params(metafunc)
    metafunc.parametrize('order,err_lim,numel',
                         [(0, 0.06, 512),
                          (1, 0.0012, 128),
                          (2, 0.001, 32)])

def myfunc(x):
    # clip input
    x = np.clip(x, -DOMAIN, +DOMAIN)
    # apply function
    return np.sin(x)

def gen_model(order=0, numel=512):
    # create mixed-signal model
    model = MixedSignalModel('model', build_dir=BUILD_DIR)
    model.add_analog_input('in_')
    model.add_analog_output('out')
    model.add_digital_input('clk')
    model.add_digital_input('rst')

    # create function
    real_func = model.make_function(myfunc, domain=[-DOMAIN, +DOMAIN],
                                    order=order, numel=numel)

    # apply function
    model.set_from_sync_func(model.out, real_func, model.in_, clk=model.clk, rst=model.rst)

    # write the model
    return model.compile_to_file(VerilogGenerator())

def test_func_sim(simulator, order, err_lim, numel):
    # generate model
    model_file = gen_model(order=order, numel=numel)

    # declare circuit
    class dut(m.Circuit):
        name = 'test_func_sim'
        io = m.IO(
            in_=fault.RealIn,
            out=fault.RealOut,
            clk=m.In(m.Clock),
            rst=m.BitIn
        )

    # create the tester
    tester = fault.Tester(dut, dut.clk)

    # initialize
    tester.poke(dut.in_, 0)
    tester.poke(dut.clk, 0)
    tester.poke(dut.rst, 1)
    tester.eval()

    # apply reset
    tester.step(2)

    # clear reset
    tester.poke(dut.rst, 0)
    tester.step(2)

    # save the outputs
    inpts = np.random.uniform(-DOMAIN, +DOMAIN, 100)
    apprx = []
    for in_ in inpts:
        tester.poke(dut.in_, in_)
        tester.step(2)
        apprx.append(tester.get_value(dut.out))

    # run the simulation
    parameters = {
        'in_range': 2*DOMAIN,
        'out_range': 2*RANGE
    }
    tester.compile_and_run(
        target='system-verilog',
        directory=BUILD_DIR,
        simulator=simulator,
        ext_srcs=[model_file, get_file('func_sim/test_func_sim.sv')],
        inc_dirs=[get_svreal_header().parent, get_msdsl_header().parent],
        parameters=parameters,
        ext_model_file=True,
        disp_type='realtime'
    )

    # evaluate the outputs
    apprx = [elem.value for elem in apprx]

    # compute the exact response to inputs
    exact = myfunc(inpts)

    # check the result
    err = np.linalg.norm(exact-apprx)
    assert err <= err_lim