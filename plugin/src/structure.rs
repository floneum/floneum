use crate::host::{State, StructureType};
use crate::plugins::main;

use crate::plugins::main::types::Structure;

use kalosm::language::kalosm_sample::StructureParser;

use wasmtime::component::__internal::async_trait;

impl State {
    pub(crate) fn get_full_structured_parser(
        &self,
        structure: &wasmtime::component::Resource<Structure>,
    ) -> Option<StructureParser> {
        match self.structures.get(structure.rep() as usize)? {
            StructureType::Num(num) => Some(StructureParser::Num {
                min: num.min,
                max: num.max,
                integer: num.integer,
            }),
            StructureType::Literal(literal) => Some(StructureParser::Literal(literal.clone())),
            StructureType::Or(or) => Some(StructureParser::Either {
                first: Box::new(self.get_full_structured_parser(&or.first)?),
                second: Box::new(self.get_full_structured_parser(&or.second)?),
            }),
            StructureType::Then(then) => Some(StructureParser::Then {
                first: Box::new(self.get_full_structured_parser(&then.first)?),
                second: Box::new(self.get_full_structured_parser(&then.second)?),
            }),
        }
    }
}

#[async_trait]
impl main::types::HostStructure for State {
    async fn num(
        &mut self,
        num: main::types::NumberParameters,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Num(num));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn literal(
        &mut self,
        literal: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Literal(literal));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn or(
        &mut self,
        or: main::types::EitherStructure,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Or(or));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn then(
        &mut self,
        then: main::types::ThenStructure,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Then(then));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Structure>) -> wasmtime::Result<()> {
        self.structures.remove(rep.rep() as usize);
        Ok(())
    }
}
